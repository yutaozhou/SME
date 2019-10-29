.. _fitresults:

Fitresults
==========

The fitresults objects is a pretty straightforward
collection of fitresults.

Note that all results are purely based on the fitting and
do not include model uncertainties from e.g. the linelist.

The given uncertainties in punc are estimated from the residual
distribution and seem to be reasonable for some parameters,
but get problematic if the parameter only affects a small number
of points (e.g. individual abundances).

Here are the fields

:maxiter: Maximum number of iterations
:chisq: Final chi square
:punc: Uncertainties of the fitparameters
:covar: covariance matrix of the fitparameters
:grad: the gradient of the cost function for each parameter
:pder: The jacobian, i.e. the derivate at each point for each parameter
:resid: The residual between observation and model
