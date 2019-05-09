#!/usr/bin/env python

import os
from pathlib import Path
from ruamel.yaml import YAML
from setuptools import setup

# Create folder structure
directory = Path("~/.sme/").expanduser()
conf = directory / "config.yaml"
atmo = directory / "atmospheres"
nlte = directory / "nlte_grids"
cache_atmo = atmo / "cache"
cache_nlte = nlte / "cache"

directory.makedir(exist_ok=True)
atmo.makedir(exist_ok=True)
nlte.makedir(exist_ok=True)
cache_atmo.makedir(exist_ok=True)
cache_nlte.makedir(exist_ok=True)


# Create config file if it does not exist
if not conf.exists():
    # Hardcode default settings?
    defaults = {
        "data.file_server": "https://sme.astro.uu.se/atmos",
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
    name="sme",
    version="0.0",
    description="Spectroscopy Made Easy",
    author="Jeff A. Valenti",
    author_email="valenti@stsci.edu",
    packages=["sme", "gui"],
    package_dir={"": "src"},
    package_data={"sme": ["dll/sme_synth.so.*", "dll/intel64_lin/*"]},
)
