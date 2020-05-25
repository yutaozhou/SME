#!/usr/bin/env python

import atexit
import json
import os
import sys
from os.path import dirname, exists, expanduser, join
from shutil import copy

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

import versioneer

import site


def binaries_directory():
    """Return the installation directory, or None"""
    if "--user" in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = "%s.%s" % (sys.version_info[0], sys.version_info[1])
        paths = (
            s % (py_version)
            for s in (
                sys.prefix + "/lib/python%s/dist-packages/",
                sys.prefix + "/lib/python%s/site-packages/",
                sys.prefix + "/local/lib/python%s/dist-packages/",
                sys.prefix + "/local/lib/python%s/site-packages/",
                "/Library/Python/%s/site-packages/",
            )
        )

    for path in paths:
        if os.path.exists(path):
            return path
    print("no installation path found", file=sys.stderr)
    return None


path = binaries_directory()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        os.system("echo Hello")
        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            from convert_fstrings import convert

            os.system(
                "echo Converting fstrings to string formats since we are using Python Version < 3.6"
            )
            convert(join(binaries_directory(), "pysme"))
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        os.system("echo World")
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            from convert_fstrings import convert

            os.system(
                "echo Converting fstrings to string formats since we are using Python Version < 3.6"
            )
            convert(join(binaries_directory(), "pysme"))
        install.run(self)


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

# Copy datafile pointers, for use in the GUI
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

cmdclass = versioneer.get_cmdclass()
cmdclass["install"] = PostInstallCommand
cmdclass["develop"] = PostDevelopCommand

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
    python_requires=">=3",
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
    # cmdclass={
    #     'develop': PostDevelopCommand,
    #     'install': PostInstallCommand,
    # },
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
