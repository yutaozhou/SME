from glob import glob
from os.path import join

import numpy as np
import io
import json
from zipfile import ZipFile

from pysme.persistence import toBaseType
from pysme.atmosphere.marcs import MarcsFile

from pysme.atmosphere.atmosphere import AtmosphereGrid

datafolder = "/DATA/MARCS/GaiaFineGrid"

# Plane Parallel
files = glob(join(datafolder, "*", "sph.t02", "????g*"))
nfiles = len(files)
npoints = MarcsFile(files[0]).ndep

grid = AtmosphereGrid(nfiles, npoints)

grid.source = "MARCS GaiaFineGrid"
grid.method = "grid"
grid.geom = "SPH"
grid.depth = "TAU"
grid.interp = "RHOX"
grid.abund_format = "H=12"
# TODO
grid.citation_info = ""


for i, file in enumerate(files):
    print(file)

    mf = MarcsFile(file)

    grid.teff[i] = mf.teff
    grid.logg[i] = mf.logg
    grid.monh[i] = mf.monh
    grid.vturb[i] = mf.vturb
    grid.lonh[i] = mf.lonh
    grid.radius[i] = mf.radius
    grid.height[i] = mf.height
    grid.opflag[i] = mf.opflag
    grid.abund[i] = mf.abund.get_pattern(raw=True, type="H=12")
    grid.temp[i] = mf.temp
    grid.rhox[i] = mf.rhox
    grid.tau[i] = mf.tau
    grid.rho[i] = mf.rho
    grid.xna[i] = mf.xna
    grid.xne[i] = mf.xne

grid.save("test.npz")
# AtmosphereGrid.load("test.npz")

