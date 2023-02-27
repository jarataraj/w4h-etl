import json

test = '{"south": -58, "north": 84, "west": 0, "east": 360}'
print(json.loads(test)["south"])


LIMITS = {"south": -58, "north": 84, "west": 0, "east": 360}
USALIMITS = {"south": 24, "north": 50, "west": 360 - 126, "east": 360 - 66}
