import os.path

from pysme.gui import plot_plotly
from pysme import sme as SME
from pysme import util
from pysme.solve import solve, synthesize_spectrum
from pysme.atmosphere import train_grid

examples_dir = os.path.dirname(os.path.realpath(__file__))
in_file = os.path.join(examples_dir, "sun_6440_grid.inp")

# Start the logging to the file
util.start_logging("training.log")

# Load your existing SME structure or create your own
sme = SME.SME_Struct.load(in_file)

train_grid(sme.atmo)
