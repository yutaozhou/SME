from os.path import dirname, join

import numpy as np

from pysme.marcs import MarcsFile


fname = join(dirname(__file__), "4000g4.5z-0.75t1")

atmo = MarcsFile(fname)
