---
title: 'PySME: Spectroscopy Made Easier'
tags:
  - Python
  - astronomy
  - spectroscopy
  - simulation
authors:
  - name: Ansgar Wehrhahn
    orcid: 0000-0002-1241-8557
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Uppsala University
   index: 1
date: 06 May 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The light spectrum of stars can tell us a lot about the characteristics of that
star, from its temperature to its chemical composition. However analysing the
spectrum is no trivial task. It requires a model of the radiative transfer
process in the stellar atmosphere to create a synthetic model of the spectrum,
which can be compared to the observations from the telescope.

`PySME` solves the 1D radiative transfer equation 

$$ I_\lambda = \int_0^\inf e^(-\tau / \mu) S_\lambda(\tau) d\tau
$$

`PySME` is a stellar model that does just that. It is based on the original
`SME` [@1996A&AS..118..595V], which was written in C, Fortran, and IDL.
`PySME` replaces the IDL component with a Python implementation, but keeps
the same C and Fortran codes, which makes it very fast.

`PySME` supports the same features as the original `SME`, including the support
for NLTE (None Linear Thermal Equlibrium) calculations.
Special care was taken that the simulation results are comparable to the original implementation.
The fitting procedure however has been replaced by a generic least-squares algorithm,
which will yield similarly good, but non-identical parameters.

The switch to Python away from IDL, brings many advantages. Foremost this
means that `PySME` is no longer bound to IDL licenses, which allows for
example for the parallel execution on servers etc. This also means that
`PySME` is now completely open source, which is a requirement for some
projects such as the upcoming 4MOST telescope [@2019Msngr.175....3D].

We hope that this will encourage old and new users to use the new `PySME`
in their research.

# Acknowledgements

We acknowledge contributions from Jeff Valenti and Nikolai Piskunov, especially in regards
to the original `SME`.

# References