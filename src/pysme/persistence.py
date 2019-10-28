import io
import logging
from zipfile import ZipFile
import json

import numpy as np

logger = logging.getLogger(__name__)


def toBaseType(value):
    if value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str):
        return str(value)

    return value


def save(filename, data, folder=""):
    """
    Create a folder structure inside a zipfile
    Add .json and .npy and .npz files with the correct names
    And subfolders for more complicated objects
    with the same layout
    Each class should have a save and a load method
    which can be used for this purpose

    Parameters
    ----------
    filename : str
        Filename of the final zipfile
    data : SME_struct
        data to save
    """
    with ZipFile(filename, "w") as file:
        saves(file, data, folder=folder)


def saves(file, data, folder=""):
    if folder != "" and folder[-1] != "/":
        folder = folder + "/"

    parameters = {}
    arrays = {}
    others = {}
    for key in data._names:
        value = getattr(data, key)
        if np.isscalar(value) or isinstance(value, dict):
            parameters[key] = value
        elif isinstance(value, (list, np.ndarray)):
            if np.size(value) > 20:
                arrays[key] = value
            else:
                parameters[key] = value
        else:
            others[key] = value

    info = json.dumps(parameters, default=toBaseType)
    file.writestr(f"{folder}info.json", info)

    for key, value in arrays.items():
        b = io.BytesIO()
        np.save(b, value)
        file.writestr(f"{folder}{key}.npy", b.getvalue())

    for key, value in others.items():
        if value is not None:
            value.save(file, f"{folder}{key}")


def load(filename, data):
    with ZipFile(filename, "r") as file:
        names = file.namelist()
        return loads(file, data, names)


def loads(file, data, names=None, folder=""):
    if folder != "" and folder[-1] != "/":
        folder = folder + "/"
    if names is None:
        names = file.namelist()

    subdirs = {}
    local = []
    for name in names:
        name_within = name[len(folder) :]
        if "/" not in name_within:
            local.append(name)
        else:
            direc, _ = name_within.split("/", 1)
            if direc not in subdirs.keys():
                subdirs[direc] = []
            subdirs[direc].append(name)

    for name in local:
        if name.endswith(".json"):
            info = file.read(name)
            info = json.loads(info)
            for key, value in info.items():
                data[key] = value
        elif name.endswith(".npy") or name.endswith(".npz"):
            b = io.BytesIO(file.read(name))
            data[name[:-4]] = np.load(b)

    for key, value in subdirs.items():
        data[key] = data[key].load(file, value, folder=folder + key)

    return data
