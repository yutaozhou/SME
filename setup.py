#!/usr/bin/env python

import os
from setuptools import setup

import versioneer
from pathlib import Path
from ruamel.yaml import YAML


# Create folder structure
directory = Path("~/.sme/").expanduser()
conf = directory / "config.yaml"
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
        "data.pointers.atmospheres": "datafiles_atmospheres.yaml",
        "data.pointers.nlte_grids": "datafiles_nlte.yaml",
    }

    # Save file to disk
    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with conf.open("w") as f:
        yaml.dump(defaults, f)


# Setup package
setup(
    name="pysme-astro",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Spectroscopy Made Easy",
    author="Ansgar Wehrhahn, Jeff A. Valenti",
    author_email="ansgar.wehrhahn@physics.uu.se, valenti@stsci.edu",
    packages=["sme", "gui"],
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "plotly",
        "pandas",
        "wget",
        "ruamel.yaml",
    ],
)
