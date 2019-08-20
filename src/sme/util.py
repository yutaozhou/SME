"""
Utility functions for SME

safe interpolation
"""

import argparse
import logging
from functools import wraps
from pathlib import Path
from platform import python_version

import numpy as np
from numpy import __version__ as npversion
from pandas import __version__ as pdversion
from scipy import __version__ as spversion
from scipy.interpolate import interp1d

from . import version as smeversion
from .sme_synth import SME_DLL

try:
    from IPython import get_ipython

    cfg = get_ipython()
    in_notebook = cfg is not None
except (AttributeError, ImportError):
    in_notebook = False


class getter:
    def __call__(self, func):
        @wraps(func)
        def fget(obj):
            value = func(obj)
            return self.fget(obj, value)

        return fget

    def fget(self, obj, value):
        raise NotImplementedError


class apply(getter):
    def __init__(self, app, allowNone=True):
        self.app = app
        self.allowNone = allowNone

    def fget(self, obj, value):
        if self.allowNone and value is None:
            return value
        if isinstance(self.app, str):
            return getattr(value, self.app)()
        else:
            return self.app(value)


class setter:
    def __call__(self, func):
        @wraps(func)
        def fset(obj, value):
            value = self.fset(obj, value)
            func(obj, value)

        return fset

    def fset(self, obj, value):
        raise NotImplementedError


class oftype(setter):
    def __init__(self, _type, allowNone=True, **kwargs):
        self._type = _type
        self.allowNone = allowNone
        self.kwargs = kwargs

    def fset(self, obj, value):
        if self.allowNone and value is None:
            return value
        elif value is None:
            raise TypeError(
                f"Expected value of type {self._type}, but got None instead"
            )
        return self._type(value, **self.kwargs)


class ofarray(setter):
    def __init__(self, dtype=float, allowNone=True):
        self.dtype = dtype
        self.allowNone = allowNone

    def fset(self, obj, value):
        if self.allowNone and value is None:
            return value
        elif value is None:
            raise TypeError(
                f"Expected value of type {self.dtype}, but got {value} instead"
            )
        arr = np.asarray(value, dtype=self.dtype)
        return np.atleast_1d(arr)


class oneof(setter):
    def __init__(self, allowed_values=()):
        self.allowed_values = allowed_values

    def fset(self, obj, value):
        if value not in self.allowed_values:
            raise ValueError(
                f"Expected one of {self.allowed_values}, but got {value} instead"
            )
        return value


class ofsize(setter):
    def __init__(self, shape, allowNone=True):
        self.shape = shape
        self.allowNone = allowNone
        if hasattr(shape, "__len__"):
            self.ndim = len(shape)
        else:
            self.ndim = 1
            self.shape = (self.shape,)

    def fset(self, obj, value):
        if self.allowNone and value is None:
            return value
        if hasattr(value, "shape"):
            ndim = len(value.shape)
            shape = value.shape
        elif hasattr(value, "__len__"):
            ndim = 1
            shape = (len(value),)
        else:
            ndim = 1
            shape = (1,)

        if ndim != self.ndim:
            raise ValueError(
                f"Expected value with {self.ndim} dimensions, but got {ndim} instead"
            )
        elif not all([i == j for i, j in zip(shape, self.shape)]):
            raise ValueError(
                f"Expected value of shape {self.shape}, but got {shape} instead"
            )
        return value


class absolute(oftype):
    def __init__(self):
        super().__init__(float)

    def fset(self, obj, value):
        value = super().fset(obj, value)
        if value is not None:
            value = abs(value)
        return value


class uppercase(oftype):
    def __init__(self):
        super().__init__(str)

    def fset(self, obj, value):
        value = super().fset(obj, value)
        if value is not None:
            value = value.upper()
        return value


class lowercase(oftype):
    def __init__(self):
        super().__init__(str)

    def fset(self, obj, value):
        value = super().fset(obj, value)
        if value is not None:
            value = value.lower()
        return value


def safe_interpolation(x_old, y_old, x_new=None, fill_value=0):
    """
    'Safe' interpolation method that should avoid
    the common pitfalls of spline interpolation

    masked arrays are compressed, i.e. only non masked entries are used
    remove NaN input in x_old and y_old
    only unique x values are used, corresponding y values are 'random'
    if all else fails, revert to linear interpolation

    Parameters
    ----------
    x_old : array of size (n,)
        x values of the data
    y_old : array of size (n,)
        y values of the data
    x_new : array of size (m, ) or None, optional
        x values of the interpolated values
        if None will return the interpolator object
        (default: None)

    Returns
    -------
    y_new: array of size (m, ) or interpolator
        if x_new was given, return the interpolated values
        otherwise return the interpolator object
    """

    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)

    mask = np.isfinite(x_old) & np.isfinite(y_old)
    x_old = x_old[mask]
    y_old = y_old[mask]

    # avoid duplicate entries in x
    # also sorts data, which allows us to use assume_sorted below
    x_old, index = np.unique(x_old, return_index=True)
    y_old = y_old[index]

    try:
        interpolator = interp1d(
            x_old,
            y_old,
            kind="cubic",
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True,
        )
    except ValueError:
        logging.warning(
            "Could not instantiate cubic spline interpolation, using linear instead"
        )
        interpolator = interp1d(
            x_old,
            y_old,
            kind="linear",
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True,
        )

    if x_new is not None:
        return interpolator(x_new)
    else:
        return interpolator


def has_logger(log_file="log.log"):
    logger = logging.getLogger()
    return len(logger.handlers) == 0


def start_logging(log_file="log.log"):
    """Start logging to log file and command line

    Parameters
    ----------
    log_file : str, optional
        name of the logging file (default: "log.log")
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing File handles
    hasStream = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Command Line output
    # only if not running in notebook
    if not in_notebook and not hasStream:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    # Log file settings
    if log_file is not None:
        log_file = Path(log_file)
        log_dir = log_file.parent
        log_dir.mkdir(exist_ok=True)
        file = logging.FileHandler(log_file)
        file.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file.setFormatter(file_formatter)
        logger.addHandler(file)

    # Turns print into logging.info
    # But messes with the debugger
    # builtins.print = lambda msg, *args, **kwargs: logging.info(msg, *args)
    logging.captureWarnings(True)

    dll = SME_DLL()
    logging.debug("----------------------")
    logging.debug("Python version: %s", python_version())
    try:
        logging.debug("SME CLib version: %s", dll.SMELibraryVersion())
    except OSError:
        logging.debug("SME CLib version: ???")
    logging.debug("PySME version: %s", smeversion.full_version)
    logging.debug("Numpy version: %s", npversion)
    logging.debug("Scipy version: %s", spversion)
    logging.debug("Pandas version: %s", pdversion)


def parse_args():
    """Parse command line arguments

    Returns
    -------
    sme : str
        filename to input sme structure
    vald : str
        filename of input linelist or None
    fitparameters : list(str)
        names of the parameters to fit, empty list if none are specified
    """

    parser = argparse.ArgumentParser(description="SME solve")
    parser.add_argument(
        "sme",
        type=str,
        help="an sme input file (either in IDL sav or Numpy npy format)",
    )
    parser.add_argument("--vald", type=str, default=None, help="the vald linelist file")
    parser.add_argument(
        "fitparameters",
        type=str,
        nargs="*",
        help="Parameters to fit, abundances are 'Mg Abund'",
    )
    args = parser.parse_args()
    return args.sme, args.vald, args.fitparameters
