""" Main entry point for an SME script """
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from gui import plot_plotly, plot_pyplot
from sme import sme as SME
from sme import util
from sme.abund import Abund
from sme.solve import synthesize_spectrum, SME_Solver
from sme.vald import ValdFile

if __name__ == "__main__":
    target = "debug"
    util.start_logging(f"{target}.log")

    # Get input files
    if len(sys.argv) > 1:
        in_file, vald_file, fitparameters = util.parse_args()
    else:
        os.chdir(os.path.dirname(__file__))
        in_file = "sun_6440_grid.inp"
        vald_file = "sun.lin"
        atmo_file = None
        fitparameters = []

    # Load files
    sme = SME.SME_Struct.load(in_file)
    sme.nlte.set_nlte("Ca")

    # sme.save("test.npz", compressed=False, for_idl=True)

    if vald_file is not None:
        vald = ValdFile(vald_file)
        sme.linelist = vald.linelist

    if atmo_file is not None:
        sme.atmo.source = atmo_file
        sme.atmo.method = "grid"

    # Choose free parameters, i.e. sme.pname
    if len(fitparameters) == 0:
        # ["teff", "logg", "monh", "abund Mg", "abund Y"]
        if sme.fitparameters is not None and len(sme.fitparameters) != 0:
            fitparameters = sme.fitparameters
        else:
            fitparameters = ["teff", "logg", "monh"]

    fitparameters = ["teff"]
    sme.nlte.set_nlte("Ca")

    # Start SME solver
    # sme = synthesize_spectrum(sme, segments=[0])
    solver = SME_Solver(filename=f"{target}.npz")
    sme = solver.solve(sme, fitparameters, segments=[0])

    try:
        # Calculate stellar age based on abundances
        solar = Abund.solar()
        y, mg = sme.abund["Y"], sme.abund["Mg"]
        sy, smg = sme.fitresults.punc["Y abund"], sme.fitresults.punc["Mg abund"]
        x = y - mg - (solar["Y"] - solar["Mg"])
        sx = np.sqrt(sy ** 2 + smg ** 2)

        # Values from paper
        a = 0.175
        sa = 0.011
        b = -0.0404
        sb = 0.0019
        age = (x - a) / b
        sigma_age = 1 / b * np.sqrt(sx ** 2 + sa ** 2 + ((x - a) / b) ** 2 * sb ** 2)
        sigma_age = abs(sigma_age)
        logging.info("Age       \t%.3f +- %.3f Gyr", age, sigma_age)

        p = np.linspace(0, 10, 1000)
        g = norm.pdf(p, loc=age, scale=sigma_age)
        # Rescale to area = 1
        area = np.sum(g * np.gradient(p))  # Cheap integral
        g *= 1 / area
        plt.plot(p, g)
        plt.xlabel("Age [Gyr]")
        plt.ylabel("Probability")
        plt.show()
    except:
        pass

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=f"{target}.html")

    # # if "synth" in sme:
    # #     plt.plot(sme.wob, sme.sob - sme.smod, label="Residual Python")
    # #     # plt.plot(sme.wave, sme.sob - orig, label="Residual IDL")
    # #     plt.legend()
    # #     plt.show()

    # mask_plot = plot_pyplot.MaskPlot(sme)
    # input("Wait a second...")
