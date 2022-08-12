import json
import pickle
import numpy as np


class JSONSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def to_json(obj, path):
    with open(path, "w") as fp:
        json.dump(obj, fp, cls=JSONSetEncoder)


def from_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def to_pickle(obj, path):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def from_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)
