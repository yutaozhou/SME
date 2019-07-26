""" Minimum working example of an SME script 

Run this from the examples directory, so that 
"""
import os.path

from gui import plot_plotly
from sme import sme as SME
from sme import util
from sme.solve import solve, synthesize_spectrum

if __name__ == "__main__":
    target = "k2_3"
    util.start_logging(f"{target}.log")

    # Put your input structure here!
    examples_dir = "/DATA/ESO_Archive/HARPS/K2-3/"
    in_file = os.path.join(examples_dir, "K2-3_red_c.ech")

    linelist = os.path.expanduser("~/Documents/IDL/SME/harps_blue.lin")

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
