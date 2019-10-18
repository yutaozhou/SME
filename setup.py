#!/usr/bin/env python

import os
import json
from setuptools import setup

import versioneer
from pathlib import Path

# Create folder structure
directory = Path("~/.sme/").expanduser()
conf = directory / "config.json"
atmo = directory / "atmospheres"
nlte = directory / "nlte_grids"
cache_atmo = atmo / "cache"
cache_nlte = nlte / "cache"

directory.mkdir(exist_ok=True)
atmo.mkdir(exist_ok=True)
nlte.mkdir(exist_ok=True)
cache_atmo.mkdir(exist_ok=True)
cache_nlte.mkdir(exist_ok=True)


# Create config file if it does not exist
if not conf.exists():
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
    with conf.open("w") as f:
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
