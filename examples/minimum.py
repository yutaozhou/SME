""" Minimum working example of an SME script """
import os
import numpy as np

from gui import plot_plotly
from sme import sme as SME
from sme import util
from sme.solve import solve, synthesize_spectrum

if __name__ == "__main__":
    target = "debug"
    util.start_logging(f"{target}.log")

    # Put your input structure here!
    os.chdir(os.path.dirname(__file__))
    in_file = "sun_6440_grid.inp"
    
    sme = SME.SME_Struct.load(in_file)
    fitparameters = ["teff", "logg", "monh"]

    # Start SME solver
    # sme = synthesize_spectrum(sme)
    sme = solve(sme, fitparameters)

    # Save results
    sme.save(f"{target}.npz")

    # Plot results
    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename=f"{target}.html")
