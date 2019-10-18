__file_ending__ = ".sme"

# Load correct version string
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Add output to the console
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

del logging

# Provide submodules to the outside
__all__ = [
    "util",
    "abund",
    "atmosphere",
    "broadening",
    "continuum_and_radial_velocity",
    "cwrapper",
    "echelle",
    "iliffe_vector",
    "integrate_flux",
    "linelist",
    "nlte",
    "sme_synth",
    "sme",
    "solve",
    "uncertainties",
    "vald",
]
