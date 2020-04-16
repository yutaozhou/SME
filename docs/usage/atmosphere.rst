.. _atmosphere:

Atmosphere
==========

For the spectral synthesis PySME needs a model atmosphere
to perform the radiative transfer in. PySME does not come
with a set of atmospheres in each distribution but instead
uses the LFS (See :ref:`lfs`) to fetch only the required
model atmosphere when run.

If you want to provide your own model atmosphere file, it should be present in `~/.sme/atmospheres/`.

Each atmosphere model file describes a grid of models, on
which we then linearly interpolate to the desired stellar parameters.
Sometimes we dare extrapolate from this grid as well, but in that case,
we always show a warnning.

Note that the atmosphere also contains a seperate set of stellar
parameters, which is usually the same as that of the sme structure,
but can be different, if for example the atmosphere is embedded, i.e.
fixed, or has not been calculated yet.

The atmopshere object has the following fields

:teff: Effective Temperature in Kelvin
:logg: Surface Gravity in log(cgs)
:monh: Metallicity relative to the individual abundances
:abund: The individual abundances (see :ref:`abund`)
:vsini: Projected Rotational velocity in km/s
:vmic: Microturbulence velocity in km/s
:vmac: Macroturbulence veclocity in km/s
:vturb: Turbulent velocity in km/s
:lonh: ? Metallicity
:source: Filename of the atmosphere grid
:depth:
    The depth scale to use for calculations.
    Either RHOX or TAU
:interp:
    The depth scale to use for interpolation.
    Either RHOX or TAU
:geom:
    The geometry of the atmopshere. Either Plane
    Parallel 'PP' or Spherical 'SPH'.
:method:
    The method to use for interpolation. Either
    'grid' for a model grid or 'embedded' if
    only a single atmosphere is given.
:rhox: 'Column density' depth scale
:tau: 'Optical depth' depth scale
:temp: Temperature profile
:xna: Number density of atoms, ions, and molecules in each depth
:xne: Number density of electrons in each depth
