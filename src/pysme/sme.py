import logging
import platform
import sys
import os.path
from datetime import datetime as dt
from functools import wraps

import numpy as np
from scipy.io import readsav

from . import __version__, __file_ending__
from . import echelle
from .abund import Abund
from .iliffe_vector import Iliffe_vector
from .linelist import LineList
from . import persistence

logger = logging.getLogger(__name__)


def this(x):
    """ This just returns the input """
    return x


def dont(x):
    raise NotImplementedError("Can't set this value directly")


def notNone(func):
    def f(value):
        return func(value) if value is not None else None

    return f


def uppercase(func):
    def f(value):
        try:
            value = value.upper()
        except AttributeError:
            pass
        return func(value)

    return f


def lowercase(func):
    def f(value):
        try:
            value = value.casefold()
        except AttributeError:
            pass
        return func(value)

    return f


def oneof(*options):
    def f(value):
        if value not in options:
            raise ValueError(f"Received {value} but expected one of {options}")
        return value

    return f


def array(shape, dtype, allow_None=True):
    def f(value):
        if value is not None:
            value = np.asarray(value, dtype=dtype)
            value = np.atleast_1d(value)
            if shape is not None:
                value = value.reshape(shape)
        elif not allow_None:
            raise ValueError(f"Received None but expected datatype {dtype}")
        return value

    return f


def vector(self, value):
    if value is None:
        return None
    elif np.isscalar(value):
        wind = [0, *np.cumsum(self.wave.sizes)] if self.wave is not None else None
        values = np.full(self.wave.size, value)
        value = Iliffe_vector(values=values, index=wind)
    elif isinstance(value, np.ndarray):
        value = np.require(value, requirements="W")
        if value.ndim == 1:
            wind = [0, *np.cumsum(self.wave.sizes)] if self.wave is not None else None
            value = Iliffe_vector(values=value, index=wind)
        else:
            value = Iliffe_vector(nseg=len(value), values=value)
    elif isinstance(value, list):
        value = Iliffe_vector(nseg=len(value), values=value)
    elif isinstance(value, Iliffe_vector):
        pass
    elif isinstance(value, np.lib.npyio.NpzFile):
        value = Iliffe_vector._load(value)
    else:
        raise TypeError("Input value is of the wrong type")

    return value


# TODO: change all fset functions to func(self, value)
# TODO: use a decorator on the class and replace __new__

# fget = lambda name, func: lambda self: func(getattr(self, f"_{name}"))
# fset = lambda name, func: lambda self, value: setattr(self, f"_{name}", func(self, value))


def fget(name, func):
    name = f"_{name}"

    def f(self):
        return func(getattr(self, name))

    return f


def fset(name, func):
    name = f"_{name}"

    def f(self, value):
        try:
            setattr(self, name, func(value))
        except TypeError:
            setattr(self, name, func(self, value))

    return f


class Collection(persistence.IPersist):
    _fields = []  # [("name", "default", str, this, "doc")]

    def __new__(cls, data=None, **kwargs):
        if data is not None and isinstance(data, cls):
            # If we get an object, try to convert it to this class
            return data
        # Add properties to the class
        for name, _, setter, getter, doc in cls._fields:
            setattr(
                cls, name, property(fget(name, getter), fset(name, setter), None, doc)
            )
        cls._names = [f[0] for f in cls._fields]

        # Initialize a new object
        self = object.__new__(cls)
        return self

    def __init__(self, data=None, **kwargs):
        if data is not None and isinstance(data, self.__class__):
            # If we got an object of the correct type we ignore it
            # That is handled in __new__
            return
        for name, default, *_ in self._fields:
            setattr(self, name, default)

        for key, value in kwargs.items():
            if key in self._names:
                if isinstance(value, bytes):
                    value = value.decode()
                elif isinstance(value, np.ndarray) and value.dtype == np.dtype("O"):
                    value = value.astype(str)
                setattr(self, key, value)

    def __getitem__(self, key):
        key = key.casefold()
        return getattr(self, key)

    def __setitem__(self, key, value):
        key = key.casefold()
        setattr(self, key, value)

    def __contains__(self, key):
        return key in dir(self) and getattr(self, key) is not None

    def citation(self, format="string"):
        return []


class Parameters(Collection):
    # fmt: off
    _fields = Collection._fields + [
        ("teff", 5770, float, this, "float: effective temperature in Kelvin"),
        ("logg", 4.0, float, this, "float: surface gravity in log10(cgs)"),
        ("abundance", Abund.solar(), Abund, this, "Abund: elemental abundances"),
        ("vmic", 0, abs, this, "float: micro turbulence in km/s"),
        ("vmac", 0, abs, this, "float: macro turbulence in km/s"),
        ("vsini", 0, abs, this, "float: projected rotational velocity in km/s"),
    ]
    # fmt: on

    def __init__(self, data=None, **kwargs):
        monh = kwargs.pop("monh", kwargs.pop("feh", 0))
        abund = kwargs.pop("abund", "empty")
        super().__init__(data=data, **kwargs)
        self.abund = Abund(monh=monh, pattern=abund, type="sme")

    @property
    def monh(self):
        """float: metallicity in log scale relative to the base abundances"""
        return self.abund.monh

    @monh.setter
    def monh(self, value):
        self.abund.monh = value

    def citation(self, format="string"):
        return self.abund.citation()


class Atmosphere(Parameters):
    """
    Atmosphere structure
    contains all information to describe the solar atmosphere
    i.e. temperature etc in the different layers
    as well as stellar parameters and abundances
    """

    # fmt: off
    _fields = Parameters._fields + [
        ("vturb", 0, abs, this, "float: turbulence velocity in km/s"),
        ("lonh", 0, float, this, "float: ?"),
        ("source", None, str, this, "str: datafile name of this data"),
        ("method", None, lowercase(oneof(None, "grid", "embedded")), this, 
            "str: whether the data source is a grid or a fixed atmosphere"),
        ("geom", None, uppercase(oneof(None, "PP", "SPH")), this,
            "str: the geometry of the atmopshere model"),
        ("radius", 0, float, this, "float: radius of the spherical model"),
        ("height", 0, float, this, "float: height of the spherical model"),
        ("opflag", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0], array(None, int), this,
            "array: opacity flags"),
        ("wlstd", 5000, float, this, "float: wavelength standard deviation"),
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


class NLTE(Collection):
    # fmt: off
    _fields = Collection._fields + [
        ("elements", [], list, this,
            "list: elements for which nlte calculations will be performed"),
        ("grids", {}, dict, this,
            "dict: nlte grid datafiles for each element"),
        ("subgrid_size", [2, 2, 2, 2], array(4, int), this,
            "array of shape (4,): defines size of nlte grid cache."
            "Each entry is for one parameter abund, teff, logg, monh"),
        ("flags", None, array(None, np.bool_), this,
            "array: contains a flag for each line, whether it was calculated in NLTE (True) or not (False)")
    ]
    # fmt: on

    _default_grids = {
        "Al": "marcs2012_Al2017.grd",
        "Fe": "marcs2012_Fe2016.grd",
        "Li": "marcs2012_Li.grd",
        "Mg": "marcs2012_Mg2016.grd",
        "Na": "marcs2012p_t1.0_Na.grd",
        "O": "marcs2012_O2015.grd",
        "Ba": "marcs2012p_t1.0_Ba.grd",
        "Ca": "marcs2012p_t1.0_Ca.grd",
        "Si": "marcs2012_SI2016.grd",
        "Ti": "marcs2012s_t2.0_Ti.grd",
    }

    def __init__(self, data=None, **kwargs):
        super().__init__(data=data)

        # Convert IDL keywords to Python
        if "nlte_elem_flags" in kwargs.keys():
            elements = kwargs["nlte_elem_flags"]
            self.elements = [Abund._elem[i] for i, j in enumerate(elements) if j == 1]

        if "nlte_subgrid_size" in kwargs.keys():
            self.subgrid_size = kwargs["nlte_subgrid_size"]

        if "nlte_grids" in kwargs:
            grids = kwargs["nlte_grids"]
            if isinstance(grids, (list, np.ndarray)):
                grids = {
                    Abund._elem[i]: name.decode()
                    for i, name in enumerate(grids)
                    if name != ""
                }
            self.grids = grids

    def set_nlte(self, element, grid=None):
        """
        Add an element to the NLTE calculations

        Parameters
        ----------
        element : str
            The abbreviation of the element to add to the NLTE calculations
        grid : str, optional
            Filename of the NLTE data grid to use for this element
            the file must be in nlte_grids directory
            Defaults to a set of "known" files for some elements
        """
        if element in self.elements:
            # Element already in NLTE
            # Change grid if given
            if grid is not None:
                self.grids[element] = grid
            return

        if grid is None:
            # Use default grid
            if element not in NLTE._default_grids.keys():
                raise ValueError(f"No default grid known for element {element}")
            grid = NLTE._default_grids[element]
            logger.info("Using default grid %s for element %s", grid, element)

        self.elements += [element]
        self.grids[element] = grid

    def remove_nlte(self, element):
        """
        Remove an element from the NLTE calculations

        Parameters
        ----------
        element : str
            Abbreviation of the element to remove from NLTE
        """
        if element not in self.elements:
            # Element not included in NLTE anyways
            return

        self.elements.remove(element)
        self.grids.pop(element)

    def citation(self, format="string"):
        citations = [self.grids[el] for el in self.elements]
        return citations


class Version(Collection):
    # fmt: off
    _fields = Collection._fields + [
        ("arch", "", str, this, "str: system architecture"),
        ("os", "", str, this, "str: operating system"),
        ("os_family", "", str, this, "str: operating system family"),
        ("os_name", "", str, this, "str: os name"),
        ("release", "", str, this, "str: python version"),
        ("build_date", "", str, this, "str: build date of the Python version used"),
        ("memory_bits", 64, int, this, "int: Platform architecture bit size (usually 32 or 64)"),
        ("host", "", str, this, "str: name of the machine that created the SME Structure")
    ]
    # fmt: on

    def __init__(self, data=None, **kwargs):
        self.update()
        super().__init__(data=data, **kwargs)

    def update(self):
        """ Update version info with current machine data """
        self.arch = platform.machine()
        self.os = sys.platform
        self.os_family = platform.system()
        self.os_name = platform.version()
        self.release = platform.python_version()
        self.build_date = platform.python_build()[1]
        self.memory_bits = int(platform.architecture()[0][:2])
        self.host = platform.node()


class Fitresults(Collection):
    # fmt: off
    _fields = Collection._fields + [
        ("maxiter", 100, int, this, "int: maximum number of iterations in the solver"),
        ("chisq", 0, float, this, "float: reduced chi-square of the solution"),
        ("uncertainties", None, array(None, float), this, "array of size(nfree,): uncertainties of the free parameters"),
        ("covariance", None, array(None, float), this, "array of size (nfree, nfree): covariance matrix"),
        ("gradient", None, array(None, float), this, "array of size (nfree,): final gradients of the free parameters on the cost function"),
        ("derivative", None, array(None, float), this, "array of size (npoints, nfree): final Jacobian of each point and each parameter"),
        ("residuals", None, array(None, float), this, "array of size (npoints,): final residuals of the fit"),
        ("punc2", None, this, this, "array: old school uncertainties")
    ]
    # fmt: on

    def clear(self):
        """ Reset all values to None """
        self.maxiter = 100
        self.chisq = 0
        self.punc = None
        self.covar = None
        self.grad = None
        self.pder = None
        self.resid = None


class SME_Structure(Parameters):
    # fmt: off
    _fields = Parameters._fields + [
        ("id", dt.now(), str, this, "str: DateTime when this structure was created"),
        ("object", "", str, this, "str: Name of the observed/simulated object"),
        ("version", __version__, this, this, "str: PySME version used to create this structure"),
        ("vrad", 0, array(None, float), this, "array of size (nseg,): radial velocity of each segment in km/s"),
        ("vrad_flag", "none", lowercase(oneof(-2, -1, 0, "none", "each", "whole", "fix")), this,
            """str: flag that determines how the radial velocity is determined

            allowed values are:
               * "none": No radial velocity correction
               * "each": Determine radial velocity for each segment individually
               * "whole": Determine one radial velocity for the whole spectrum
            """),
        ("cscale", 1, array(None, float), this,
            """array of size (nseg, ndegree): Continumm polynomial coefficients for each wavelength segment
            The x coordinates of each polynomial are chosen so that x = 0, at the first wavelength point,
            i.e. x is shifted by wave[segment][0]
            """),
        ("cscale_flag", "none", lowercase(oneof(-3, -2, -1, 0, 1, 2, "none", "fix", "constant", "linear", "quadratic")), this,
            """str: Flag that describes how to correct for the continuum

            allowed values are:
                * "none": No continuum correction
                * "fix": Use whatever continuum scale has been set, but don't change it
                * "constant": Zeroth order polynomial, i.e. scale everything by a factor
                * "linear": First order polynomial, i.e. approximate continuum by a straight line
                * "quadratic": Second order polynomial, i.e. approximate continuum by a quadratic polynomial
            """),
        ("normalize_by_continuum", True, bool, this,
            "bool: Whether to normalize the synthetic spectrum by the synthetic continuum spectrum or not"),
        ("gam6", 1, float, this, "float: van der Waals scaling factor"),
        ("h2broad", True, bool, this, "bool: Whether to use H2 broadening or not"),
        ("accwi", 0.003, float, this,
            "float: minimum accuracy for linear spectrum interpolation vs. wavelength. Values below 1e-4 are not meaningful."),
        ("accrt", 0.001, float, this,
            "float: minimum accuracy for synthethized spectrum at wavelength grid points in sme.wint. Values below 1e-4 are not meaningful."),
        ("iptype", None, lowercase(oneof(None, "gauss", "sinc", "table")), this, "str: instrumental broadening type"),
        ("ipres", 0, float, this, "float: Instrumental resolution for instrumental broadening"),
        ("ip_x", None, this, this, "array: Instrumental broadening table in x direction"),
        ("ip_y", None, this, this, "array: Instrumental broadening table in y direction"),
        ("mu", np.geomspace(0.01, 1, 7), array(None, float), this,
            """array of size (nmu,): Mu values to calculate radiative transfer at
            mu values describe the distance from the center of the stellar disk to the edge
            with mu = cos(theta), where theta is the angle of the observation,
            i.e. mu = 1 at the center of the disk and 0 at the edge
            """),
        ("wran", None, this, this,
            "array of size (nseg, 2): beginning and end wavelength points of each segment"),
        ("wave", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): wavelength"),
        ("spec", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): observed spectrum"),
        ("uncs", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): uncertainties of the observed spectrum"),
        ("mask", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): mask defining good and bad points for the fit"),
        ("synth", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): synthetic spectrum"),
        ("cont", None, vector, this,
            "Iliffe_vector of shape (nseg, ...): continuum intensities"),
        ("linelist", None, LineList, this, "LineList: spectral line information"),
        ("fitparameters", [], list, this, "list: parameters to fit"),
        ("atmo", None, Atmosphere, this, "Atmosphere: model atmosphere data"),
        ("nlte", None, NLTE, this, "NLTE: nlte calculation data"),
        ("system_info", None, Version, this,
            "Version: information about the host system running the calculation for debugging"),
    ]
    # fmt: on

    #:dict(str, int): Mask value specifier used in mob
    mask_values = {"bad": 0, "line": 1, "continuum": 2}

    def __init__(self, data=None, **kwargs):
        self.wind = kwargs.get("wind", None)
        if self.wind is not None:
            self.wind = np.array([0, *(self.wind + 1)])

        super().__init__(data=data, **kwargs)
        del self.wind

        self.spec = kwargs.get("sob", None)
        self.uncs = kwargs.get("uob", None)
        self.mask = kwargs.get("mob", None)
        self.synth = kwargs.get("smod", None)

        self.object = kwargs.get("obs_name", "")
        try:
            self.linelist = LineList(**kwargs)
        except (KeyError, AttributeError):
            # TODO ignore the warning during loading of data
            logger.warning("No or incomplete linelist data present")

        # Parse free parameters into one list
        pname = kwargs.get("pname", [])
        glob_free = kwargs.get("glob_free", [])
        ab_free = kwargs.get("ab_free", [])
        if len(ab_free) != 0:
            ab_free = [f"abund {el}" for i, el in zip(ab_free, Abund._elem) if i == 1]
        fitparameters = np.concatenate((pname, glob_free, ab_free)).astype("U")
        #:array of size (nfree): Names of the free parameters
        self.fitparameters = np.unique(fitparameters)

        self.fitresults = Fitresults(
            maxiter=kwargs.pop("maxiter", 0),
            chisq=kwargs.pop("chisq", 0),
            uncertainties=kwargs.pop("punc", None),
            covar=kwargs.pop("covar", None),
        )

        self.normalize_by_continuum = kwargs.get("cscale_flag", "") != "fix"

        self.system_info = Version(**kwargs.get("idlver", {}))
        self.system_info.update()
        self.atmo = Atmosphere(
            **kwargs.get("atmo", {}),
            abund=kwargs.get("abund", "empty"),
            monh=kwargs.get("monh", kwargs.get("feh", 0)),
        )
        self.nlte = NLTE(**kwargs.get("nlte", {}))

        # Apply final conversions from IDL to Python version
        if "wave" in self:
            self.__convert_cscale__()

    def __getitem__(self, key):
        assert isinstance(key, str), "Key must be of type string"
        key = key.casefold()

        if key.startswith("abund "):
            element = key[5:].strip()
            element = element.capitalize()
            return self.abund[element]
        if key.startswith("linelist "):
            _, idx, field = key[8:].split(" ", 2)
            idx = int(idx)
            return self.linelist[field][idx]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, str), "Key must be of type string"
        key = key.casefold()

        if key.startswith("abund "):
            element = key[5:].strip()
            element = element.capitalize()
            self.abund[element] = value
        elif key.startswith("linelist "):
            _, idx, field = key[8:].split(" ", 2)
            idx = int(idx)
            self.linelist[field][idx] = value
        else:
            super().__setitem__(key, value)

    # Additional constraints on fields
    @property
    def _wran(self):
        if self.wave is not None:
            nseg = self.wave.shape[0]
            values = np.zeros((nseg, 2))
            for i in range(nseg):
                if self.wave[i] is not None and len(self.wave[i]) >= 2:
                    values[i] = [self.wave[i][0], self.wave[i][-1]]
                else:
                    values[i] = self.__wran[i]
            return values
        return self.__wran

    @_wran.setter
    def _wran(self, value):
        try:
            if self.wave is not None:
                logger.warning(
                    "The wavelength range is overriden by the existing wavelength grid"
                )
        except:
            pass
        self.__wran = np.atleast_2d(value) if value is not None else None

    @property
    def _vrad(self):
        """array of size (nseg,): Radial velocity in km/s for each wavelength region"""
        nseg = self.nseg if self.nseg is not None else 1

        if self.__vrad is None:
            return np.zeros(nseg)

        if self.vrad_flag == "none":
            return np.zeros(nseg)
        else:
            nseg = self.__vrad.shape[0]
            if nseg == self.nseg:
                return self.__vrad

            rv = np.zeros(self.nseg)
            rv[:nseg] = self.__vrad[:nseg]
            rv[nseg:] = self.__vrad[-1]
            return rv

        return self.__vrad

    @_vrad.setter
    def _vrad(self, value):
        self.__vrad = np.atleast_1d(value) if value is not None else None

    @property
    def _vrad_flag(self):
        return self.__vrad_flag

    @_vrad_flag.setter
    def _vrad_flag(self, value):
        if isinstance(value, (int, np.integer)):
            value = {-2: "none", -1: "whole", 0: "each"}[value]
        self.__vrad_flag = value

    @property
    def _cscale(self):
        """array of size (nseg, ndegree): Continumm polynomial coefficients for each wavelength segment
        The x coordinates of each polynomial are chosen so that x = 0, at the first wavelength point,
        i.e. x is shifted by wave[segment][0]
        """
        nseg = self.nseg if self.nseg is not None else 1

        if self.__cscale is None:
            cs = np.zeros((nseg, self.cscale_degree + 1))
            cs[:, -1] = 1
            return cs

        if self.cscale_flag == "none":
            return np.ones((nseg, 1))

        ndeg = self.cscale_degree + 1
        ns, nd = self.__cscale.shape

        if nd == ndeg and ns == nseg:
            return self.__cscale

        cs = np.zeros((nseg, ndeg))
        cs[:, -1] = 1
        if nd > ndeg:
            cs[:ns, :] = self.__cscale[:ns, -ndeg:]
        elif nd < ndeg:
            cs[:ns, -nd:] = self.__cscale[:ns, :]
        else:
            cs[:ns, :] = self.__cscale[:ns, :]

        cs[ns:, -1] = self.__cscale[-1, -1]

        return cs

    @_cscale.setter
    def _cscale(self, value):
        self.__cscale = np.atleast_2d(value) if value is not None else None

    @property
    def _cscale_flag(self):
        return self.__cscale_flag

    @_cscale_flag.setter
    def _cscale_flag(self, value):
        if isinstance(value, (int, np.integer)):
            value = {
                -3: "none",
                -2: "fix",
                -1: "fix",
                0: "constant",
                1: "linear",
                2: "quadratic",
            }[value]

        self.__cscale_flag = value

    @property
    def _mu(self):
        return self.__mu

    @_mu.setter
    def _mu(self, value):
        if np.any(value < 0):
            raise ValueError("All values must be positive")
        if np.any(value > 1):
            raise ValueError("All values must be smaller or equal to 1")
        self.__mu = value

    # Additional properties
    @property
    def nseg(self):
        """int: Number of wavelength segments """
        if self.wran is None:
            return 0
        else:
            return len(self.wran)

    @property
    def nmu(self):
        return self.mu.size

    @property
    def mask_good(self):
        if self.mask is None:
            return None
        return self.mask != self.mask_values["bad"]

    @property
    def mask_bad(self):
        if self.mask is None:
            return None
        return self.mask == self.mask_values["bad"]

    @property
    def cscale_degree(self):
        """int: Polynomial degree of the continuum as determined by cscale_flag """
        if self.cscale_flag == "constant":
            return 0
        if self.cscale_flag == "linear":
            return 1
        if self.cscale_flag == "quadratic":
            return 2
        if self.cscale_flag == "fix":
            # Use the underying element to avoid a loop
            if self.__cscale is not None:
                return self.__cscale.shape[1] - 1
            else:
                return 0
        if self.cscale_flag == "none":
            return 0
        raise ValueError("This should never happen")

    @property
    def atomic(self):
        """array of size (nlines, 8): Atomic linelist data, usually passed to the C library
        Use sme.linelist instead for other purposes """
        if self.linelist is None:
            return None
        return self.linelist.atomic

    @property
    def species(self):
        """array of size (nlines,): Names of the species of each spectral line """
        if self.linelist is None:
            return None
        return self.linelist.species

    def __convert_cscale__(self):
        """
        Convert IDL SME continuum scale to regular polynomial coefficients
        Uses Taylor series approximation, as IDL version used the inverse of the continuum
        """
        wave = self.wave
        self.cscale = np.require(self.cscale, requirements="W")

        if self.cscale_flag == "linear":
            for i in range(len(self.cscale)):
                c, d = self.cscale[i]
                a, b = max(wave[i]), min(wave[i])
                c0 = (a - b) * (c - d) / (a * c - b * d) ** 2
                c1 = (a - b) / (a * c - b * d)

                # Shift zero point to first wavelength of the segment
                c1 += c0 * self.spec[i][0]

                self.cscale[i] = [c0, c1]
        elif self.cscale_flag == "fix":
            self.cscale = self.cscale / np.sqrt(2)
        elif self.cscale_flag == "constant":
            self.cscale = np.sqrt(1 / self.cscale)

    def citation(self, format="string"):
        citation = ["SME citation", "PySME citation"]
        citation += self.atmo.citation(format=format)
        citation += self.nlte.citation(format=format)
        return citation

    def save(self, filename, compressed=False):
        # __file_ending__ = ".sme"
        if not filename.endswith(__file_ending__):
            filename = filename + __file_ending__
        persistence.save(filename, self)

    @staticmethod
    def load(filename):
        """
        Load SME data from disk

        Currently supported file formats:
            * ".npy": Numpy save file of an SME_Struct
            * ".sav", ".inp", ".out": IDL save file with an sme structure
            * ".ech": Echelle file from (Py)REDUCE

        Parameters
        ----------
        filename : str, optional
            name of the file to load (default: 'sme.npy')

        Returns
        -------
        sme : SME_Struct
            Loaded SME structure

        Raises
        ------
        ValueError
            If the file format extension is not recognized
        """
        logger.info("Loading SME file %s", filename)
        ext = os.path.splitext(filename)[1]
        if ext == ".sme":
            s = SME_Structure()
            return persistence.load(filename, s)
        elif ext == ".npy":
            # Numpy Save file
            s = np.load(filename, allow_pickle=True)
            return np.atleast_1d(s)[0]
        elif ext == ".npz":
            s = np.load(filename, allow_pickle=True)
            return s["sme"][()]
        elif ext in [".sav", ".out", ".inp"]:
            # IDL save file (from SME)
            s = readsav(filename)["sme"]

            def unfold(obj):
                if isinstance(obj, bytes):
                    return obj.decode()
                elif isinstance(obj, np.recarray):
                    return {
                        name.casefold(): unfold(obj[name][0])
                        for name in obj.dtype.names
                    }
                return obj

            s = unfold(s)
            return SME_Structure(**s)
        elif ext == ".ech":
            # Echelle file (from REDUCE)
            ech = echelle.read(filename)
            s = SME_Structure()
            s.wave = [np.ma.compressed(w) for w in ech.wave]
            s.spec = [np.ma.compressed(s) for s in ech.spec]
            s.uncs = [np.ma.compressed(s) for s in ech.sig]

            for i, w in enumerate(s.wave):
                sort = np.argsort(w)
                s.wave[i] = w[sort]
                s.spec[i] = s.spec[i][sort]
                s.uncs[i] = s.uncs[i][sort]

            s.mask = [np.full(i.size, 1) for i in s.spec]
            s.mask[s.spec == 0] = SME_Structure.mask_values["bad"]
            s.wran = [[w[0], w[-1]] for w in s.wave]
            s.abund = Abund.solar()
            try:
                s.object = ech.head["OBJECT"]
            except KeyError:
                pass
            return s
        else:
            options = [".npy", ".sav", ".out", ".inp", ".ech"]
            raise ValueError(
                f"File format not recognised, expected one of {options} but got {ext}"
            )
