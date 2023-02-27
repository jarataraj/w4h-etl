import functions_framework

import xarray as xr
import numpy as np
import pandas as pd
import thermofeel_fork.thermofeel as thermofeel
from utils import (
    load,
    open_dataset,
    erbs_ufunc,
    update_cloud_data_with,
    copy_cloud_data_to,
    Status,
)

# PLOTTING:
from cartopy import crs as ccrs, feature as cfeature
import matplotlib.pyplot as plt

# I/O
import re
import os
import json
from pymongo import MongoClient, ReplaceOne

DATA_VARS = [
    "tmp2m",
    "ugrd10m",
    "vgrd10m",
    "dpt2m",
    "dswrfsfc",
    "dlwrfsfc",
    "uswrfsfc",
    "ulwrfsfc",
]

MONGODB_URL = os.environ.get("MONGODB_URL")
CLOUD_MEDIA_STORAGE_BASE_URL = os.environ.get("CLOUD_STORAGE_BASE_URL")
CLOUD_MEDIA_STORAGE_ACCESS_KEY = os.environ.get("CLOUD_STORAGE_ACCESS_KEY")
LIMITS = json.loads(os.environ.get("LIMITS"))


@functions_framework.http
def main(request):
    source_url = request.json["url"]
    print(f"DATA_PIPELINE: new ETL run for {source_url}")
    try:
        source = open_dataset(source_url)
    except Exception as ex:
        print(ex)
        return ("DATA_PIPELINE: Unable to open dataset", 500)
    ds = xr.merge(
        [
            source[var]
            # exlude hour 0 due to missing solar data
            .isel(time=slice(1, 121)).sel(
                lat=slice(LIMITS["south"], LIMITS["north"]),
                lon=slice(LIMITS["west"], LIMITS["east"]),
            )
            for var in DATA_VARS
        ]
    )
    # ====== Mean Radiant Temperature ======
    for var in ["uswrfsfc", "ulwrfsfc", "dlwrfsfc", "dswrfsfc"]:
        try:
            load(ds[var])
        except Exception:
            return (f"DATA_PIPELINE: Unable to load {var} data", 500)

    # ------ calc avg_solar_cza ------
    # note: would normally divide by period, but period is already 1
    ds["avg_solar_cza"] = xr.concat(
        [
            xr.apply_ufunc(
                thermofeel.calculate_cos_solar_zenith_angle_integrated,
                ds.lat,
                ds.lon,
                time.dt.year,
                time.dt.month,
                time.dt.day,
                time.dt.hour,
                0,
                1,
            )
            for time in ds.time
        ],
        "time",
    )

    # ------ calc direct_normal_irradiance ------
    # erbs(dswrfsfc, avg_solar_zenith, day_of_year)
    (
        ds["direct_normal_irradiance"],
        ds["diffuse_horizontal_irradiance"],
    ) = xr.apply_ufunc(
        erbs_ufunc,
        ds["dswrfsfc"],
        np.arccos(ds["avg_solar_cza"]),
        ds["time"].dt.dayofyear,
        output_core_dims=[(), ()],
    )

    # ------ calc mean_radiant_temp ------
    # calculate_mean_radiant_temperature(ssrd, ssr, dsrp, strd, fdir, strr, cossza)
    # fdir = dni, dsrp = solar direct through horizontal plane
    ds["mean_radiant_temp"] = xr.apply_ufunc(
        thermofeel.calculate_mean_radiant_temperature,
        ds.dswrfsfc,
        ds.dswrfsfc - ds.uswrfsfc,
        ds.dswrfsfc - ds.diffuse_horizontal_irradiance,
        ds.dlwrfsfc,
        ds.direct_normal_irradiance,
        ds.dlwrfsfc - ds.ulwrfsfc,
        ds.avg_solar_cza,
    )

    # ------ free memory -------
    # delete mean radiant tempeture components to save memory
    ds = ds.drop_vars(
        [
            "dswrfsfc",
            "dlwrfsfc",
            "uswrfsfc",
            "ulwrfsfc",
            "avg_solar_cza",
            "direct_normal_irradiance",
            "diffuse_horizontal_irradiance",
        ]
    )

    # ====== Wind Speed ======
    # load wind speed components
    for var in ["ugrd10m", "vgrd10m"]:
        try:
            load(var)
        except Exception:
            return (f"DATA_PIPELINE: Unable to load {var} data", 500)
    # calc wind speed from components
    ds["wind_speed"] = np.hypot(ds["ugrd10m"], ds["ugrd10m"])

    # delete wind speed components to save memory
    ds = ds.drop_vars(["vgrd10m", "ugrd10m"])

    # ====== UTCI, WBGT ======
    for var in ["tmp2m", "dpt2m"]:
        try:
            load(ds[var])
        except Exception:
            return (f"DATA_PIPELINE: Unable to load {var} data", 500)
    # calculate_utci(t2_k, va_ms, mrt_k, ehPa, td_k)
    ds["utci"] = xr.apply_ufunc(
        thermofeel.calculate_utci,
        ds.tmp2m,
        ds.wind_speed,
        ds.mean_radiant_temp,
        None,
        ds.dpt2m,
    )
    # def calculate_wbgt(t_k, mrt, va, td, p=None)
    ds["wbgt"] = xr.apply_ufunc(
        thermofeel.calculate_wbgt,
        ds.tmp2m,
        ds.mean_radiant_temp,
        ds.wind_speed,
        ds.dpt2m,
    )
    # free memory
    ds = ds.drop_vars(["tmp2m", "dpt2m", "wind_speed", "mean_radiant_temp"])

    # download old data
    copy_cloud_data_to("previous_w4h_data.nc")

    # ====== Times ======
    # earliest global chart that can be updated with new data
    earliest_global_chart_to_update = pd.Timestamp(ds.time[0].item()).floor("d")
    # haa = hour-angle adjusted
    utc_now = pd.Timestamp.utcnow()
    # earliest haa utc-labeled date
    # i.e. what is the current utc-labelled date for longitudes of hour-angle -11
    earliest_current_haa_utc_date = (
        utc_now.tz_localize(None) - pd.Timedelta(11, "h")
    ).floor("d")
    # earliest date accessible to global charts (i.e. earliest utc-labeled 'yesterday')
    earliest_global_chart_date = earliest_current_haa_utc_date - pd.Timedelta(1, "d")
    # earliest time necessary for charting (subtract 12 since data is shifted forwards up to 12 forwards)
    earliest_global_chart_data = earliest_global_chart_to_update - pd.Timedelta(12, "h")
    # start of local day will always be within 25 hours of the current time (24 + 1 for daylight savings time)
    earliest_forecast_time = (
        (utc_now - pd.Timedelta(25, "h")).floor("d").tz_localize(None)
    )
    earliest_necessary_data = min(earliest_forecast_time, earliest_global_chart_data)

    # merge old with new
    ds.combine_first(
        xr.open_dataset("previous_w4h_data.nc").sel(time=slice(earliest_necessary_data))
    )
    # remove old data file to free memory
    os.remove("previous_w4h_data.nc")

    # ====== Data Upload =====

    # ------ Encoding ------
    ds["time_offset"] = ((ds.time - ds.time[0]) / 3600000000000).astype(np.int16)
    # encode UTCI
    ds["encoded_temp_times"] = ((ds.utci + 100) * 10).round().astype(np.int32)
    # encode WBGT
    ds.encoded_temp_times = ds.encoded_temp_times * 2000 + (
        (ds.wbgt + 100) * 10
    ).round().astype(np.int32)
    # encode time
    ds.encoded_temp_times = ds.encoded_temp_times * 120 + ds.time_offset

    # free memory
    ds = ds.drop_vars(["time_offset"])

    def createRecord(forecast_start, lat, lon, da):
        forecast = da.sel(dict(lat=lat, lon=lon))
        dic = forecast.to_dict()
        strlat = format(dic["coords"]["lat"]["data"], ".2f")
        strlon = format(dic["coords"]["lon"]["data"], ".2f")
        tempTimesEncoded = dic["data"]
        record = dict(
            _id=strlat + "," + strlon,
            forecastStart=forecast_start,
            tempTimesEncoded=tempTimesEncoded,
        )
        return record

    forecast_start = ds.time[0].to_dict()["data"]
    near_land = xr.open_dataset("near_land.nc").near_land

    uploads = [
        createRecord(forecast_start, lat, lon, ds.encoded_temp_times)
        for lat in ds.encoded_temp_times.lat
        for lon in ds.encoded_temp_times.lon
        if near_land.sel(lat=lat, lon=lon)
    ]
    requests = [
        ReplaceOne({"_id": record["_id"]}, record, upsert=True) for record in uploads
    ]

    db = MongoClient(MONGODB_URL).testTempsDB
    db.forecasts2.bulk_write(requests)
    status = Status(db)
    status.set("latestSuccessfulUpdateSource", source_url)

    # free memory
    ds = ds.drop_vars(["encoded_temp_times", "wbgt"])

    # save data to cloud storage
    ds.to_netcdf("new_w4h_data.nc")
    update_cloud_data_with("new_w4h_data.nc")
    os.remove("new_w4h_data.nc")

    # ===== Charting =====
    # select UTCI data to chart and add cyclic when using worldwide data
    if ds.UTCI.sel(lon=0).any():
        ds = xr.concat([ds.UTCI, ds.UTCI.sel(lon=0).assign_coords(lon=360)], dim="lon")
    else:
        ds = ds.UTCI

    # remove reference to charts older than earliest "yesterday" from status
    global_charts_removed = []
    for date in status.status["globalCharts"].keys():
        if pd.Timestamp(date) < earliest_global_chart_date:
            global_charts_removed.append(date)
            status.delete(f"globalCharts.{date}")
    if global_charts_removed:
        print(f"removed from status.globalCharts: {global_charts_removed}")

    # ------ shift data according to hour angle ------
    # haa = hour-angle-adjusted
    hour_angle = (ds.lon / 15).round().astype(int)
    utc_hour_angle = xr.where(hour_angle > 12, hour_angle - 24, hour_angle)
    haa_ds = xr.full_like(ds, np.nan)
    for offset in np.unique(utc_hour_angle):
        haa_ds = haa_ds.combine_first(
            ds.where(utc_hour_angle == offset).shift(time=offset)
        )

    # prepare base chart
    colors = [
        "#004adb",
        "#306cde",
        "#468de0",
        "#5aadde",
        "#75cdd6",
        "#b3e8b6",
        "#ffde98",
        "#fcad6e",
        "#f27946",
        "#e43a20",
    ]
    divisions = [-40, -27, -13, 0, 9, 26, 32, 38, 46]

    def quantize(da, divisions):
        quantized = xr.where(da >= divisions[-1], len(divisions), np.nan)
        for i, division in enumerate(divisions):
            quantized = quantized.combine_first(xr.where(da < division, i, np.nan))
        return quantized

    projection = ccrs.Miller(central_longitude=11)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_frame_on(False)
    ax.autoscale_view()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#444")
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="#444")

    source_info = re.search(
        r"\/gfs(\d{4})(\d{2})(\d{2}).*(\d{2}z)$", source_url
    ).groups()
    source_date = "-".join(source_info[:3])
    source_time = source_info[-1]

    for date in np.unique(haa_ds.time.dt.date):
        data = (haa_ds.where(haa_ds.time.dt.date == date, drop=True)).utci.dropna(
            "time"
        )
        if len(data.time) < 24:
            continue
        for (vertex, data) in [("highs", data.max("time")), ("lows", data.min("time"))]:
            quantized_data = quantize(data, divisions)
            contour_set = quantized_data.plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                add_labels=False,
                add_colorbar=False,
                vmin=-0.5,
                vmax=9.5,
                levels=11,
                colors=colors,
            )
            # create image file
            file_name = f"{date}Z_utci_{vertex}_from_gfs_data_up_to_{source_date}_{source_time}.png"
            fig.savefig(f"{file_name}", dpi=100, pad_inches=0, bbox_inches="tight")

            # clear plotted contours for next chart
            for contour in contour_set.collections:
                contour.remove()

            # upload file to cloud storage
            upload_url = f"{CLOUD_MEDIA_STORAGE_BASE_URL}/{date}Z/{file_name}"
            with open(f"{file_name}", "rb") as upload:
                res = requests.put(
                    upload_url,
                    headers={"AccessKey": CLOUD_MEDIA_STORAGE_ACCESS_KEY},
                    data=upload,
                )
            if res.status_code != 201:
                print(f"error uploading {file_name}")
            else:
                # update status
                status.set(f"globalCharts.{date}", f"{source_date}_{source_time}")
            # delete image file
            os.remove(f"{file_name}")
        # upon ETL completion
        return f"DATA_PIPELINE: completed using: {source_url}"
