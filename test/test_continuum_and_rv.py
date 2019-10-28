# TODO implement continuum and radial velocity tests

from os.path import dirname

import pytest
import numpy as np

from pysme.sme import SME_Struct
from pysme.continuum_and_radial_velocity import determine_rv_and_cont


def test_match_both(testcase1):
    sme, x_syn, y_syn, rv = testcase1
    # Nothing should change
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"

    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert rvel == 0
    assert cscale == [1]

    sme.vrad_flag = "each"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert cscale == [1]

    sme.vrad_flag = "whole"
    rvel, cscale = determine_rv_and_cont(sme, range(sme.nseg), x_syn, y_syn)
    assert rvel == 0
    assert cscale == [1]

    rvel, cscale = determine_rv_and_cont(
        sme, range(sme.nseg), x_syn, y_syn, use_whole_spectrum=True
    )

    assert np.allclose(rvel, rv, atol=1)
    assert cscale == [1]

    sme.vrad_flag = "each"
    sme.cscale_flag = "constant"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [1], atol=1e-2)

    sme.cscale_flag = "linear"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 1], atol=1e-2)

    sme.cscale_flag = "quadratic"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 0, 1], atol=1e-2)


def test_nomask(testcase1):
    sme, x_syn, y_syn, rv = testcase1
    sme.cscale_flag = "linear"
    sme.vrad_flag = "each"

    mask = np.copy(sme.mask[0])
    sme.mask = 1

    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 1], atol=1e-2)

    sme.mask = 0
    with pytest.warns(UserWarning):
        rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert rvel == 0
    assert cscale == [1]
