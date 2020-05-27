#!/usr/bin/env python

import json
import os
from os.path import dirname, exists, expanduser, join
from shutil import copy
import platform
import tarfile

from setuptools import setup

import wget

import versioneer


def download_libsme(lib=None):
    if lib is None:
        lib = dirname(__file__)
    # Download compiled library from github releases
    print("Downloading and installing the latest libsme version for this system")
    aliases = {"Linux": "manylinux2010", "Windows": "win64", "Darwin": "osx"}

    system = platform.system()
    try:
        system = aliases[system]
    except KeyError:
        raise KeyError(
            "Could not find the associated compiled library for this system {}. Either compile it yourself and place it in src/pysme/ or open an issue on Github"
        )

    github_releases_url = "https://github.com/AWehrhahn/SMElib/releases/latest/download"
    github_releases_fname = "smelib_{system}.tar.gz".format(system=system)
    url = github_releases_url + "/" + github_releases_fname
    loc = join(lib, "src/pysme")
    fname = join(loc, github_releases_fname)

    if os.path.exists(fname):
        os.remove(fname)

    print("Downloading file %s" % url)
    os.makedirs(loc, exist_ok=True)
    wget.download(url, out=loc)

    with tarfile.open(fname) as tar:
        tar.extractall(loc)
    os.remove(fname)


cmdclass = versioneer.get_cmdclass()
_build_py = cmdclass["build_py"]

class BuildWrapper(_build_py):
    def run(self):
        download_libsme(self.build_lib)
        _build_py.run(self)

cmdclass["build_py"] = BuildWrapper

# Create folder structure for config files
print("Set up the configuration files for PySME")
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
else:
    print("Configuration file already exists")

# Copy datafile pointers, for use in the GUI
print("Copy references to datafiles to config directory")
copy(
    join(dirname(__file__), "src/pysme/datafiles_atmospheres.json"),
    expanduser("~/.sme/datafiles_atmospheres.json"),
)
copy(
    join(dirname(__file__), "src/pysme/datafiles_nlte.json"),
    expanduser("~/.sme/datafiles_nlte.json"),
)

# TODO: Have smelib compiled before distribution
with open(join(dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

# Setup package
setup(
    name="pysme-astro",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Spectroscopy Made Easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ansgar Wehrhahn, Jeff A. Valenti",
    author_email="ansgar.wehrhahn@physics.uu.se, valenti@stsci.edu",
    packages=["pysme", "pysme.gui", "pysme.atmosphere", "pysme.linelist"],
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "astropy",
        "pandas",
        "wget",
        "requests",
        "tqdm",
        "colorlog",
        "emcee",
        "pybtex",
    ],
    url="https://github.com/AWehrhahn/SME/",
    project_urls={
        "Bug Tracker": "https://github.com/AWehrhahn/SME/issues",
        "Documentation": "https://pysme-astro.readthedocs.io/en/latest/",
        "Source Code": "https://github.com/AWehrhahn/SME/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
