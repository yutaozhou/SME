""" Handles reading and interpolation of atmopshere (grid) data """
import itertools
import logging

import numpy as np
from scipy.io import readsav

from ..data_structure import (
    CollectionFactory,
    Collection,
    absolute,
    this,
    asfloat,
    asstr,
    lowercase,
    uppercase,
    oneof,
    array,
)
from ..abund import Abund

logger = logging.getLogger(__name__)


class AtmosphereError(RuntimeError):
    """ Something went wrong with the atmosphere interpolation """


@CollectionFactory
class Atmosphere(Collection):
    """
    Atmosphere structure
    contains all information to describe the solar atmosphere
    i.e. temperature etc in the different layers
    as well as stellar parameters and abundances
    """

    # fmt: off
    _fields = [
        ("teff", 5770, asfloat, this, "float: effective temperature in Kelvin"),
        ("logg", 4.0, asfloat, this, "float: surface gravity in log10(cgs)"),
        ("abund", Abund.solar(), this, this, "Abund: elemental abundances"),
        ("vturb", 0, absolute, this, "float: turbulence velocity in km/s"),
        ("lonh", 0, asfloat, this, "float: ?"),
        ("source", None, asstr, this, "str: datafile name of this data"),
        ("method", None, lowercase(oneof(None, "grid", "embedded")), this, 
            "str: whether the data source is a grid or a fixed atmosphere"),
        ("geom", None, uppercase(oneof(None, "PP", "SPH")), this,
            "str: the geometry of the atmopshere model"),
        ("radius", 0, asfloat, this, "float: radius of the spherical model"),
        ("height", 0, asfloat, this, "float: height of the spherical model"),
        ("opflag", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0], array(20, int), this,
            "array of size (20,): opacity flags"),
        ("wlstd", 5000, asfloat, this, "float: wavelength standard deviation"),
        ("depth", None, uppercase(oneof(None, "RHOX", "TAU")), this,
            "str: flag that determines whether to use RHOX or TAU for calculations"),
        ("interp", None, uppercase(oneof(None, "RHOX", "TAU")), this,
            "str: flag that determines whether to use RHOX or TAU for interpolation"),
        ("rhox", None, array(None, "f8"), this,
            "array: mass column density"),
        ("tau", None, array(None, "f8"), this, 
            "array: continuum optical depth"),
        ("temp", None, array(None, "f8"), this,
            "array: temperature profile in Kelvin"),
        ("rho", None, array(None, "f8"), this,
            "array: density profile"),
        ("xna", None, array(None, "f8"), this,
            "array: number density of atoms in 1/cm**3"),
        ("xne", None, array(None, "f8"), this,
            "array: number density of electrons in 1/cm**3")
    ]
    # fmt: on

    @property
    def names(self):
        return self._names

    @property
    def dtype(self):
        obj = lambda: None
        obj.names = [n.upper() for n in self.names]
        return obj

    @property
    def ndep(self):
        scale = self.temp
        if scale is not None:
            return scale.shape[0]
        return None

    @ndep.setter
    def ndep(self, value):
        pass

    def citation(self, format="string"):
        if self.source is None:
            return []
        # TODO Get the data from the file
        return [self.source]


@CollectionFactory
class AtmosphereGrid(Collection):
    # fmt: off
    _fields = [
        ("teff", None, array("ngrids", float), this, "array of shape (ngrids,): effective temperature in Kelvin"),
        ("logg", None, array("ngrids", float), this, "array of shape (ngrids,): surface gravity in log10(cgs)"),
        ("abund", None, array("ngrids,99", float), this, "array of shape (ngrids, 99): elemental abundances"),
        ("vturb", None, array("ngrids", float), this, "array of shape (ngrids,): turbulence velocity in km/s"),
        ("lonh", 0, array("ngrids", float), this, "float: ?"),
        ("source", None, asstr, this, "str: datafile name of this data"),
        ("method", None, lowercase(oneof(None, "grid", "embedded")), this, 
            "str: whether the data source is a grid or a fixed atmosphere"),
        ("geom", None, uppercase(oneof(None, "PP", "SPH")), this,
            "str: the geometry of the atmopshere model"),
        ("radius", 0, array("ngrids", float), this, "float: radius of the spherical model"),
        ("height", 0, array("ngrids", float), this, "float: height of the spherical model"),
        ("opflag", None, array("ngrids,20", int), this,
            "array of size (ngrids, 20): opacity flags"),
        ("wlstd", None, array("ngrids", float), this, "float: wavelength standard deviation"),
        ("depth", None, uppercase(oneof(None, "RHOX", "TAU")), this,
            "str: flag that determines whether to use RHOX or TAU for calculations"),
        ("interp", None, uppercase(oneof(None, "RHOX", "TAU")), this,
            "str: flag that determines whether to use RHOX or TAU for interpolation"),
        ("rhox", None, array("ngrids,npoints", "f8"), this,
            "array: mass column density"),
        ("tau", None, array("ngrids,npoints", "f8"), this, 
            "array: continuum optical depth"),
        ("temp", None, array("ngrids,npoints", "f8"), this,
            "array: temperature profile in Kelvin"),
        ("rho", None, array("ngrids,npoints", "f8"), this,
            "array: density profile"),
        ("xna", None, array("ngrids,npoints", "f8"), this,
            "array: number density of atoms in 1/cm**3"),
        ("xne", None, array("ngrids,npoints", "f8"), this,
            "array: number density of electrons in 1/cm**3")
    ]
    # fmt: on


class sav_file(AtmosphereGrid):
    """ IDL savefile atmosphere grid """

    def __init__(self, filename, lfs_atmo):
        super().__init__()
        self.lfs_atmo = lfs_atmo

        path = lfs_atmo.get(filename)
        data = readsav(path)

        self.method = "grid"
        self.npoints = data["atmo_grid_maxdep"]
        self.ngrids = data["atmo_grid_natmo"]
        self.source = filename
        self.citation = data["atmo_grid_intro"]
        self.citation = [d.decode() for d in self.citation]
        self.citation = "".join(self.citation)

        atmo_grid = data["atmo_grid"]

        if "RADIUS" in atmo_grid.dtype.names:
            self.geom = "SPH"
        else:
            self.geom = "PP"

        # Scalar Parameters (one per atmosphere)
        self.teff = atmo_grid["teff"]
        self.logg = atmo_grid["logg"]
        self.monh = atmo_grid["monh"]
        self.vturb = atmo_grid["vturb"]
        self.lonh = atmo_grid["lonh"]
        # Vector Parameters (one array per atmosphere)
        self.rhox = np.stack(atmo_grid["rhox"])
        self.tau = np.stack(atmo_grid["tau"])
        self.temp = np.stack(atmo_grid["temp"])
        self.rho = np.stack(atmo_grid["rho"])
        self.xne = np.stack(atmo_grid["xne"])
        self.xna = np.stack(atmo_grid["xna"])
        self.abund = np.stack(atmo_grid["abund"])


class krz_file(Atmosphere):
    """ Read .krz atmosphere files """

    def __init__(self, filename):
        super().__init__()
        self.source = filename
        self.method = "embedded"
        self.load(filename)

    def load(self, filename):
        """
        Load data from disk

        Parameters
        ----------
        filename : str
            name of the file to load
        """
        # TODO: this only works for some krz files
        # 1..2 lines header
        # 3 line opacity
        # 4..13 elemntal abundances
        # 14.. Table data for each layer
        #    Rhox Temp XNE XNA RHO

        with open(filename, "r") as file:
            header1 = file.readline()
            header2 = file.readline()
            opacity = file.readline()
            abund = [file.readline() for _ in range(10)]
            table = file.readlines()

            # Parse header
            # vturb
        i = header1.find("VTURB")
        self.vturb = float(header1[i + 5 : i + 9])
        # L/H, metalicity
        i = header1.find("L/H")
        self.lonh = float(header1[i + 3 :])

        k = len("T EFF=")
        i = header2.find("T EFF=")
        j = header2.find("GRAV=", i + k)
        self.teff = float(header2[i + k : j])

        i = j
        k = len("GRAV=")
        j = header2.find("MODEL TYPE=", i + k)
        self.logg = float(header2[i + k : j])

        i, k = j, len("MODEL TYPE=")
        j = header2.find("WLSTD=", i + k)
        model_type_key = {0: "rhox", 1: "tau", 3: "sph"}
        self.model_type = int(header2[i + k : j])
        self.depth = model_type_key[self.model_type]
        self.geom = "pp"

        i = j
        k = len("WLSTD=")
        self.wlstd = float(header2[i + k :])

        # parse opacity
        i = opacity.find("-")
        opacity = opacity[:i].split()
        self.opflag = np.array([int(k) for k in opacity])

        # parse abundance
        pattern = np.genfromtxt(abund).flatten()[:-1]
        pattern[1] = 10 ** pattern[1]
        self.abund = Abund(monh=0, pattern=pattern, type="sme")

        # parse table
        self.table = np.genfromtxt(table, delimiter=",", usecols=(0, 1, 2, 3, 4))
        self.rhox = self.table[:, 0]
        self.temp = self.table[:, 1]
        self.xne = self.table[:, 2]
        self.xna = self.table[:, 3]
        self.rho = self.table[:, 4]
