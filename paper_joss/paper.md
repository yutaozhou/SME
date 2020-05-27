---
title: 'PySME: Spectroscopy Made Easier'
tags:
  - Python
  - astronomy
  - stars
  - spectroscopy
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
<!-- Easy to understand introduction stellar synthesis modelling -->
The spectrum of a star can tell us a lot about its characteristics, from its 
surface temperature to its chemical composition. However extracting those parameters
from the spectrum is no trivial task. We approach this problem by first 
creating a synthetic spectrum, based on a set of stellar parameters and a
radiative transfer model. The parameters can then be varied until the best fit
between model and observation is found.

<!-- Introducing PySME, what is it, where does it come from -->
`PySME` is a stellar model that does just that. It is based on the original
"Spectroscopy Made Easy" (`SME`) code [@sme], which was written in C, Fortran, and IDL.
`PySME` replaces the IDL component with a Python implementation, but keeps
the same C and Fortran codes, which makes it very fast. This also means that `PySME`
can take advantage of any developments that are made to the original libraries.

<!-- PySME advantages over IDL SME, and why you should switch -->
The switch to Python away from IDL, brings many advantages. Foremost this
means that the execution of `PySME` is no longer bound to IDL licenses, which allows for
example for the parallel execution on servers etc. This also means that
`PySME` is now completely open source, which is a requirement for some
projects, such as the upcoming 4MOST telescope [@4most].
To make the transition easier it is possible to import existing `SME` configuration
files into `PySME` and immediately start the analysis.

<!-- Short summary how it works -->
To model the spectrum `PySME` calculates the 1D radiative transfer equation
$$ I_\lambda = \int_0^\infty e^{-\tau / \mu} S_\lambda(\tau) d\tau $$
where $\tau$ is the optical depth, $\mu$ is the cosine of the incidence angle,
and $S_\lambda$ is the source function (i.e. Planck's law in Local Thermal Equlibrium (LTE)),
for each wavelength point in the spectrum.
The model requires five sets of input parameters to work, most of which can also be used
variable parameters that should be determined.

The five sets are:
\begin{itemize}
 \item Stellar Parameters (temperature, surface gravity, metallicity, etc.)
 \item Model Atmosphere Grid (or a single model atmosphere)
 \item Linelist, i.e. parameters for each spectral line
 \item Elemental abundances of the star
 \item Wavelength range(s) to calculate
 \item (optional) NLTE departure coefficients
\end{itemize}

To make the analysis easier, `PySME` provides access to several model atmosphere grids from
`MARCS` [@marcs] and `ATLAS9` [@atlas9]. Also avalaible are several sets of solar abunances for
the elemental composition of the star. The linelist meanwhile can most easily be obtained from `VALD` [@vald]
using the "extract stellar", which can then directly be read by `PySME`.

<!-- Describe most important features -->
`PySME` supports the same features as the original `SME`, including the support
for NLTE (Non LTE) calculations. Similar to the atmosphere grids, `PySME` provides access to a 
variety of NLTE grids for common elements.

Special care was taken that the synthesis results are identical (within numerical uncertainties) to the original implementation. The fitting procedure however has been replaced by a generic least-squares algorithm,
which will yield similarly good, but non-identical parameters.

![An example of the synthetic spectrum compared to the observed spectrum of HD-????.\label{fig:example}](example.png)

`PySME` also comes with a convenient user interface, to manage the input data. This interface provides similar
functionality to the GUI provided with the original `SME`, but has been completely reworked to be more
interactive and user friendly.

<!-- Please use my work -->
We hope that this will encourage old and new users to use the new `PySME`
in their research.

# Acknowledgements
<!-- Thank Jeff and Nikolai, do I need to mention anyone else? -->
We acknowledge contributions from Jeff Valenti and Nikolai Piskunov, especially in regards
to the original `SME`.

# References