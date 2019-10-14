__file_ending__ = ".sme"

from . import (
    util,
    abund,
    atmosphere,
    broadening,
    continuum_and_radial_velocity,
    cwrapper,
    echelle,
    iliffe_vector,
    integrate_flux,
    linelist,
    nlte,
    sme_synth,
    sme,
    solve,
    uncertainties,
    vald,
)

util.start_logging(None)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

