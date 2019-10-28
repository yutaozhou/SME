#!/usr/bin/env python

import json
import os
from os.path import exists, expanduser, join

import versioneer
from setuptools import setup

# Create folder structure
directory = expanduser("~/.sme/")
conf = join(directory, "config.json")
atmo = join(directory, "atmospheres")
nlte = join(directory, "nlte_grids")
cache_atmo = join(atmo, "cache")
cache_nlte = join(nlte, "cache")

os.makedirs(directory, exist_ok=True)
os.makedirs(atmo, exist_ok=True)
os.makedirs(nlte, exist_ok=True)
os.makedirs(cache_atmo, exist_ok=True)
os.makedirs(cache_nlte, exist_ok=True)


# Create config file if it does not exist
if not exists(conf):
    # Hardcode default settings?
    defaults = {
        "data.file_server": "http://sme.astro.uu.se/atmos",
        "data.atmospheres": "~/.sme/atmospheres",
        "data.nlte_grids": "~/.sme/nlte_grids",
        "data.cache.atmospheres": "~/.sme/atmospheres/cache",
        "data.cache.nlte_grids": "~/.sme/nlte_grids/cache",
        "data.pointers.atmospheres": "datafiles_atmospheres.json",
        "data.pointers.nlte_grids": "datafiles_nlte.json",
    }

    # Save file to disk
    with open(conf, "w") as f:
        json.dump(defaults, f)

# Setup package
setup(
    name="pysme-astro",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Spectroscopy Made Easy",
    author="Ansgar Wehrhahn, Jeff A. Valenti",
    author_email="ansgar.wehrhahn@physics.uu.se, valenti@stsci.edu",
    packages=["pysme"],
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "plotly", "pandas", "wget"],
)
