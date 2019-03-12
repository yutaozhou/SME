"""
Handle the Yaml configuration file
At the moment it is only used for the LargeFileStorage
"""

from pathlib import Path

from ruamel.yaml import YAML


def _requires_load(func):
    def func_new(self, *args, **kwargs):
        if self._cfg is None:
            self.load()
        return func(self, *args, **kwargs)

    return func_new


class Config:
    def __init__(self, fname="~/.sme/config.yaml"):
        self.filename = fname
        self._yaml = YAML(typ="safe")
        self._yaml.default_flow_style = False
        self._cfg = None

    @property
    def filename(self):
        return str(self._filename)

    @filename.setter
    def filename(self, value):
        self._filename = Path(value).expanduser()

    @_requires_load
    def __getitem__(self, key):
        return self._cfg[key]

    @_requires_load
    def __setitem__(self, key, value):
        self._cfg[key] = value

    def load(self):
        with self._filename.open("r") as f:
            self._cfg = self._yaml.load(f)
        return self._cfg

    @_requires_load
    def save(self):
        with self._filename.open("w") as f:
            self._yaml.dump(self._cfg, f)
