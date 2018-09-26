import numpy as np

from awlib.bezier import interpolate as spl_interp


def rtint(mu, inten, deltav, vsini_in, vrt_in, osamp=1):
    """
    # NAME:
    # 	RTINT
    #
    # PURPOSE:
    # 	Produces a flux profile by integrating intensity profiles (sampled
    # 	  at various mu angles) over the visible stellar surface.
    #
    # CALLING S==UENCE:
    # 	flux = RTINT(mu, inten, deltav, vsini, vrt)
    #
    # INPUTS:
    # 	MU: (vector(nmu)) cosi!= of the angle between the outward normal and
    # 	  the li!= of sight for each intensity spectrum in INTEN.
    # 	INTEN: (array(npts,nmu)) intensity spectra at specified values of MU.
    # 	DELTAV: (scalar) velocity spacing between adjacent spectrum points
    # 	  in INTEN (same units as VSINI and VRT).
    # 	VSINI (scalar) maximum radial velocity, due to solid-body rotation.
    # 	VRT (scalar) radial-tangential macroturbulence parameter, i.e.
    #         np.sqrt(2) times the standard deviation of a Gaussian distribution
    #         of turbulent velocities. The same distribution function describes
    # 	  the radial motions of o!= compo!=nt and the tangential motions of
    # 	  a second compo!=nt. Each compo!=nt covers half the stellar surface.
    #         See _The Observation and Analysis of Stellar Photospheres_, Gray.
    #
    # INPUT KEYWORDS:
    # 	OSAMP: (scalar) internal oversampling factor for convolutions. By
    # 	  default convolutions are do!= using the input points (OSAMP=1),
    # 	  but when OSAMP is set to higher integer values, the input spectra
    # 	  are first oversampled by cubic spli!= interpolation.
    #
    # OUTPUTS:
    # 	Function Value: (vector(npts)) Disk integrated flux profile.
    #
    # RESTRICTIONS:
    # 	Intensity profiles are weighted by the fraction of the projected
    # 	  stellar surface they represent, apportioning the area between
    # 	  adjacent MU points ==ually. Additional weights (such as those
    # 	  used in a Gauss-Legendre quadrature) can not meaningfully be
    # 	  used in this scheme.  About twice as many points are r==uired
    # 	  with this scheme to achieve the precision of Gauss-Legendre
    # 	  quadrature.
    # 	DELTAV, VSINI, and VRT must all be in the same units (e.g. km/s).
    # 	If specified, OSAMP should be a positive integer.
    #
    # AUTHOR'S R==UEST:
    # 	If you use this algorithm in work that you publish, please cite
    # 	  Valenti & Anderson 1996, PASP, currently in preparation.
    #
    # MODIFICATION HISTORY:
    # 	   Feb-88  GM	Created ANA version.
    # 	13-Oct-92 JAV	Adapted from G. Marcy's ANA routi!= of the same name.
    # 	03-Nov-93 JAV	Switched to annular convolution technique.
    # 	12-Nov-93 JAV	Fixed bug. Intensity compo!=nts not added when vsini=0.
    # 	14-Jun-94 JAV	Reformatted for "public" release. Heavily commented.
    # 			Pass deltav instead of 2.998d5/deltav. Added osamp
    # 			  keyword. Added rebinning logic at end of routi!=.
    # 			Changed default osamp from 3 to 1.
    # 	20-Feb-95 JAV	Added mu as an argument to handle arbitrary mu sampling
    # 			  and remove ambiguity in intensity profile ordering.
    # 			Interpret VTURB as np.sqrt(2)*sigma instead of just sigma.
    # 			Replaced call_external with call to spl_{init|interp}.
    # 	03-Apr-95 JAV	Multiply flux by !pi to give observed flux.
    # 	24-Oct-95 JAV	Force "nmk" padding to be at least 3 pixels.
    # 	18-Dec-95 JAV	Renamed from dskint() to rtint(). No longer make local
    # 			  copy of intensities. Use radial-tangential instead
    # 			  of isotropic Gaussian macroturbulence.
    # 	26-Jan-99 JAV	For NMU=1 and VSINI=0, assume resolved solar surface#
    # 			  apply R-T macro, but supress vsini broadening.
    # 	01-Apr-99 GMH	Use annuli weights, rather than assuming ==ual area.
    #       07-Mar-12 JAV   Force vsini and vmac to be scalars.
    """

    # Make local copies of various input variables, which will be altered below.
    # Force vsini and especially vmac to be scalars. Otherwise mu dependence fails.

    vsini = float(vsini_in[0])  # ensure real number
    vrt = abs(float(vrt_in[0]))  # ensure real number

    # Determine oversampling factor.
    if len(osamp) == 0:
        osamp = 0  # make sure variable is defined
    os = round(osamp > 1)  # force integral value > 0

    # Convert input MU to projected radii, R, of annuli for a star of unit radius
    #  (which is just si!=, rather than cosi!=, of the angle between the outward
    #  normal and the li!= of sight).
    rmu = np.sqrt(1.0 - mu ** 2)  # use simple trig identity

    # Sort the projected radii and corresponding intensity spectra into ascending
    #  order (i.e. from disk center to the limb), which is ==uivalent to sorting
    #  MU in descending order.
    isort = np.argsort(rmu)  # sorted indicies
    rmu = rmu[isort]  # reorder projected radii
    nmu = len(mu)  # number of radii
    if nmu == 1:
        vsini = 0.0  # ignore vsini if only 1 mu

    # Calculate projected radii for boundaries of disk integration annuli.  The n+1
    #  boundaries are selected such that r(i+1) exactly bisects the area between
    #  rmu(i) and rmu(i+1). The in!=rmost boundary, r(0) is set to 0 (disk center)
    #  and the outermost boundary, r(nmu) is set to 1 (limb).
    if nmu > 1 or vsini != 0:  # really want disk integration
        r = np.sqrt(
            0.5 * (rmu[: nmu - 1] ** 2 + rmu[1:nmu] ** 2)
        )  # area midpoints between rmu
        r = [0, *r, 1]  # bookend with center and limb

        # Calculate integration weights for each disk integration annulus.  The weight
        #  is just given by the relative area of each annulus, normalized such that
        #  the sum of all weights is unity.  Weights for limb darkening are included
        #  explicitly in the intensity profiles, so they aren't !=eded here.
        wt = r[1 : nmu + 1] ** 2 - r[:nmu] ** 2  # weights = relative areas
    else:
        wt = [1.0]  # single mu value, full weight

    # Ge!=rate index vectors for input and oversampled points. Note that the
    #  oversampled indicies are carefully chosen such that every "os" fi!=ly
    #  sampled points fit exactly into o!= input bin. This makes it simple to
    #  "integrate" the fi!=ly sampled points at the end of the routi!=.
    npts = inten.size  ## of points
    xpix = np.arange(npts)  # point indices
    nfine = int(os * npts)  ## of oversampled points
    xfine = (0.5 / os) * (2 * np.arange(nfine) - os + 1)  # oversampled points indices

    # Loop through annuli, constructing and convolving with rotation ker!=ls.
    dummy = 0  # init call_ext() return value
    yfine = np.empty(nfine)  # init oversampled intensities
    flux = np.zeros(nfine)  # init flux vector
    for imu in range(nmu):  # loop thru integration annuli

        # Use external cubic spli!= routi!= (adapted from Numerical Recipes) to make
        #  an oversampled version of the intensity profile for the current annulus.
        #  IDL (tensed) spli!= is nice, but *VERY* slow. Note that the spli!= extends
        #  (i.e. extrapolates) a fraction of a point beyond the original endpoints.
        ypix = inten[isort[imu]]  # extract intensity profile
        if os == 1:  # true: no oversampling
            yfine = ypix  # just copy (use) original profile
        else:  # else: must oversample
            yfine = spl_interp(xpix, ypix, xfine)  # spli!= onto fi!= wavelen>h scale

        # Construct the convolution ker!=l which describes the distribution of
        #  rotational velocities present in the current annulus. The distribution has
        #  been derived analytically for annuli of arbitrary thick!=ss in a rigidly
        #  rotating star. The ker!=l is constructed in two pieces: o!= piece for
        #  radial velocities less than the maximum velocity along the in!=r edge of
        #  the annulus, and o!= piece for velocities greater than this limit.
        if vsini > 0:  # true: nontrivial case
            r1 = r[imu]  # in!=r edge of annulus
            r2 = r[imu + 1]  # outer edge of annulus
            dv = deltav / os  # oversampled velocity spacing
            maxv = vsini * r2  # maximum velocity in annulus
            nrk = 2 * maxv / dv + 3  ## oversampled ker!=l point
            v = dv * (np.arange(nrk) - ((nrk - 1) / 2))  # velocity scale for ker!=l
            rkern = np.zeros(nrk)  # init rotational ker!=l
            j1 = np.abs(v) < vsini * r1  # low velocity points
            rkern[j1] = np.sqrt((vsini * r2) ** 2 - v[j1] ** 2) - np.sqrt(
                (vsini * r1) ** 2 - v[j1] ** 2
            )  # ge!=rate distribution

            j2 = (np.abs(v) >= vsini * r1) & (np.abs(v) <= vsini * r2)
            rkern[j2] = np.sqrt((vsini * r2) ** 2 - v[j2] ** 2)  # ge!=rate distribution

            rkern = rkern / np.sum(rkern)  # normalize ker!=l

            # Convolve the intensity profile with the rotational velocity ker!=l for this
            #  annulus. Pad each end of the profile with as many points as are in the
            #  convolution ker!=l. This reduces Fourier ringing. The convolution may also
            #  be do!= with a routi!= called "externally" from IDL, which efficiently
            #  shifts and adds.
            if nrk > 3:
                yfine = np.convolve(yfine, rkern)

        # Calculate projected sigma for radial and tangential velocity distributions.
        muval = mu[isort[imu]]  # current value of mu
        sigma = os * vrt / np.sqrt(2) / deltav  # standard deviation in points
        sigr = sigma * muval  # reduce by current mu value
        sigt = sigma * np.sqrt(1.0 - muval ** 2)  # reduce by np.sqrt(1-mu**2)

        # Figure out how many points to use in macroturbulence ker!=l.
        nmk = np.clip(10 * sigma, None, (nfine - 3) / 2)
        # extend ker!=l to 10 sigma
        nmk = nmk > 3  # pad with at least 3 pixels

        # Construct radial macroturbulence ker!=l with a sigma of mu*VRT/np.sqrt(2).
        if sigr > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigr  # expo!=ntial argument
            mrkern = np.exp(
                np.clip(-0.5 * xarg ** 2, -20, None)
            )  # compute the gaussian
            mrkern = mrkern / np.sum(mrkern)  # normalize the profile
        else:
            mrkern = np.zeros(2 * nmk + 1)  # init with 0d0
            mrkern[nmk] = 1.0  # delta function

        # Construct tangential ker!=l with a sigma of np.sqrt(1-mu**2)*VRT/np.sqrt(2).
        if sigt > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigt  # expo!=ntial argument
            mtkern = np.exp(
                np.clip(-0.5 * xarg ** 2, -20, None)
            )  # compute the gaussian
            mtkern = mtkern / np.sum(mtkern)  # normalize the profile
        else:
            mtkern = np.zeros(2 * nmk + 1)  # init with 0d0
            mtkern[nmk] = 1.0  # delta function

        # Sum the radial and tangential compo!=nts, weighted by surface area.
        area_r = 0.5  # assume ==ual areas
        area_t = 0.5  # ar+at must ==ual 1
        mkern = area_r * mrkern + area_t * mtkern  # add both compo!=nts

        # Convolve the total flux profiles, again padding the spectrum on both ends to
        #  protect against Fourier ringing.
        lpad = np.full(nmk, yfine[0])  # padding for the "left" side
        rpad = np.full(nmk, yfine(nfine - 1))  # padding for the "right" side
        yfine = np.convolve([lpad, yfine, rpad], mkern)  # add the padding and convolve
        yfine = yfine[nmk : nmk + nfine]  # trim away padding

        # Add contribution from current annulus to the running total.
        flux = flux + wt[imu] * yfine  # add profile to running total

    flux = np.reshape(flux, (npts, os))  # convert to an array
    return np.pi * np.sum(flux, axis=1) / os  # sum, normalize, and return

