""" Minimum working example of an SME script 

Run this from the examples directory, so that 
"""
import os.path

from gui import plot_plotly
from sme import sme as SME
from sme import util
from sme.solve import solve, synthesize_spectrum

if __name__ == "__main__":
    target = "sun"
    util.start_logging(f"{target}.log")

    # Put your input structure here!
    examples_dir = os.path.dirname(os.path.realpath(__file__))
    in_file = os.path.join(examples_dir, "sun_6440_grid.inp")

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
