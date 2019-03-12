#!/usr/bin/env python

import os
from ruamel.yaml import YAML
from setuptools import setup

# Create folder structure
directory = os.path.expanduser("~/.sme/")
conf = os.path.join(directory, "config.yaml")
atmo = os.path.join(directory, "atmospheres")
nlte = os.path.join(directory, "nlte_grids")
cache_atmo = os.path.join(atmo, "cache")
cache_nlte = os.path.join(nlte, "cache")

os.makedirs(directory, exist_ok=True)
os.makedirs(atmo, exist_ok=True)
os.makedirs(nlte, exist_ok=True)
os.makedirs(cache_atmo, exist_ok=True)
os.makedirs(cache_nlte, exist_ok=True)


# Create config file if it does not exist
if not os.path.exists(conf):
    # Hardcode default settings?
    defaults = {
        "data.file_server": "localhost",
        "data.atmospheres": "~/.sme/atmospheres",
        "data.nlte_grids": "~/.sme/nlte_grids",
        "data.cache.atmospheres": "~/.sme/atmospheres/cache",
        "data.cache.nlte_grids": "~/.sme/nlte_grids/cache",
        "data.pointers.atmospheres": "datafiles_atmospheres.yaml",
        "data.pointers.nlte_grids": "datafiles_nlte.yaml",
    }

    # Save file to disk
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(conf, "w") as f:
        yaml.dump(defaults, f)


# Setup package
setup(
    name="sme",
    version="0.0",
    description="Spectroscopy Made Easy",
    author="Jeff A. Valenti",
    author_email="valenti@stsci.edu",
    packages=["sme", "gui"],
    package_dir={"": "src"},
    package_data={"sme": ["dll/sme_synth.so.*", "dll/intel64_lin/*"]},
)
