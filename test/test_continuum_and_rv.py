# TODO implement continuum and radial velocity tests

from os.path import dirname

import pytest
import numpy as np

from pysme.sme import SME_Structure as SME_Struct
from pysme.continuum_and_radial_velocity import determine_rv_and_cont


def test_match_both(testcase1):
    sme, x_syn, y_syn, rv = testcase1

    # Fix random results of the MCMC
    np.random.seed(0)

    vrad_options = ["fix", "none", "each", "whole"]
    cscale_options = ["fix", "none", "constant", "linear"]  # quadratic

    for voption in vrad_options:
        for coption in cscale_options:
            for segment in [[0], range(sme.nseg)]:
                sme.vrad_flag = voption
                sme.cscale_flag = coption
                sme.vrad = None
                sme.cscale = None

                vrad, vunc, cscale, cunc = determine_rv_and_cont(
                    sme, segment, x_syn[segment], y_syn[segment]
                )

                assert vrad is not None
                assert vunc is not None
                assert cscale is not None
                assert cunc is not None

                assert vrad.ndim == 1
                assert vrad.shape[0] == len(segment)
                assert vunc.ndim == 2
                assert vunc.shape[0] == len(segment)
                assert vunc.shape[1] == 2
                assert cscale.ndim == 2
                assert cscale.shape[0] == len(segment)
                assert cscale.shape[1] == sme.cscale_degree + 1
                assert cunc.ndim == 3
                assert cunc.shape[0] == len(segment)
                assert cunc.shape[1] == sme.cscale_degree + 1
                assert cunc.shape[2] == 2

                if voption in ["none", "fix"]:
                    assert np.all(vrad == 0)
                else:
                    assert np.allclose(vrad, rv, atol=1)

                assert np.allclose(cscale[:, -1], 1, atol=1e-1)
                assert np.allclose(cscale[:, :-1], 0, atol=1e-1)


def test_nomask(testcase1):
    sme, x_syn, y_syn, rv = testcase1
    sme.cscale_flag = "constant"
    sme.vrad_flag = "each"

    sme.mask = 0
    with pytest.warns(UserWarning):
        rvel, vunc, cscale, cunc = determine_rv_and_cont(sme, 0, x_syn[0], y_syn[0])

