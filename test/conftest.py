import pytest

from os.path import dirname

# TODO create various kinds of default sme structures
# then run test on all of the relevant ones
from sme.sme import SME_Struct
from sme.vald import ValdFile


@pytest.fixture
def sme_empty():
    sme = SME_Struct()
    return sme


@pytest.fixture
def sme_2segments():
    cwd = dirname(__file__)

    sme = SME_Struct()
    sme.teff = 5000
    sme.logg = 4.4
    sme.vmic = 1
    sme.vmac = 1
    sme.vsini = 1
    sme.set_abund(0, "asplund2009", "")
    sme.linelist = ValdFile(f"{cwd}/testcase1.lin").linelist
    sme.atmo.source = "marcs2012p_t2.0.sav"
    sme.atmo.method = "grid"

    sme.wran = [[6550, 6560], [6560, 6574]]

    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    return sme
