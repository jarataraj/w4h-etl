import os
import xarray as xr
from pvlib.irradiance import erbs
from google.cloud import storage
from retry import retry

W4H_CLOUD_DATA_BUCKET_NAME = os.environ.get("W4H_CLOUD_DATA_BUCKET_NAME")


@retry(RuntimeError)
def load(dataArray):
    return dataArray.load()


@retry(Exception)
def open_dataset(url):
    return xr.open_dataset(url)


# wrapper necessary for apply_ufunc, which expects return value = array or tuple of arrays
def erbs_ufunc(ghi, zenith, doy):
    erbs_results = erbs(ghi, zenith, doy)
    return (erbs_results["dni"], erbs_results["dhi"])


# Modified from: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
def update_cloud_data_with(source_file_name):
    storage_client = storage.Client(project="weather-for-humans")
    bucket = storage_client.bucket(W4H_CLOUD_DATA_BUCKET_NAME)
    blob = bucket.blob("w4h_data.nc")
    # no `generation_match_precondition` set so that existing data is overwritten
    blob.upload_from_filename(source_file_name)


# Modified from: [https://cloud.google.com/storage/docs/downloading-objects#storage-download-object-python]
def copy_cloud_data_to(destination_file_name):
    storage_client = storage.Client(project="weather-for-humans")
    bucket = storage_client.bucket(W4H_CLOUD_DATA_BUCKET_NAME)
    blob = bucket.get_blob("w4h_data.nc")
    blob.download_to_filename(destination_file_name)


# ------ Status ------
class Status:
    def __init__(self, db):
        self.db = db
        self.status = None
        self.fetch()

    def set(self, field, value):
        self.db.status.update_one({"_id": "status"}, {"$set": {field: value}})

    def fetch(self):
        self.status = self.db.status.find_one({"_id": "status"})
        return self.status

    def delete(self, field):
        self.db.status.update_one({"_id": "status"}, {"$unset": {field: ""}})
