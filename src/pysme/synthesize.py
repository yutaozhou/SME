import logging
import warnings

import numpy as np
from tqdm import tqdm
from scipy.constants import speed_of_light

from . import broadening
from .atmosphere import interp_atmo_grid
from .continuum_and_radial_velocity import match_rv_continuum
from .integrate_flux import integrate_flux
from .large_file_storage import setup_lfs
from .iliffe_vector import Iliffe_vector
from .nlte import update_nlte_coefficients
from .sme_synth import SME_DLL
from .util import safe_interpolation

logger = logging.getLogger(__name__)

clight = speed_of_light * 1e-3  # km/s


class Synthesizer:
    def __init__(self, config=None, lfs_atmo=None, lfs_nlte=None, dll=None):
        self.config, self.lfs_atmo, self.lfs_nlte = setup_lfs(
            config, lfs_atmo, lfs_nlte
        )
        self.dll = dll if dll is not None else SME_DLL()

    def get_atmosphere(self, sme):
        """
        Return an atmosphere based on specification in an SME structure

        sme.atmo.method defines mode of action:
            "grid"
                interpolate on atmosphere grid
            "embedded"
                No change
            "routine"
                calls sme.atmo.source(sme, atmo)

        Parameters
        ---------
            sme : SME_Struct
                sme structure with sme.atmo = atmosphere specification

        Returns
        -------
        sme : SME_Struct
            sme structure with updated sme.atmo
        """

        # Handle atmosphere grid or user routine.
        atmo = sme.atmo
        func = self

        if hasattr(func, "msdi_save"):
            msdi_save = func.msdi_save
            prev_msdi = func.prev_msdi
        else:
            msdi_save = None
            prev_msdi = None

        if atmo.method == "grid":
            reload = msdi_save is None or atmo.source != prev_msdi[1]
            atmo = interp_atmo_grid(
                sme.teff, sme.logg, sme.monh, sme.atmo, self.lfs_atmo, reload=reload
            )
            prev_msdi = [atmo.method, atmo.source, atmo.depth, atmo.interp]
            setattr(func, "prev_msdi", prev_msdi)
            setattr(func, "msdi_save", True)
        elif atmo.method == "routine":
            atmo = atmo.source(sme, atmo)
        elif atmo.method == "embedded":
            # atmo structure already extracted in sme_main
            pass
        else:
            raise AttributeError("Source must be 'grid', 'routine', or 'embedded'")

        sme.atmo = atmo
        return sme

    def get_wavelengthrange(self, wran, vrad, vsini):
        """
        Determine wavelengthrange that needs to be calculated
        to include all lines within velocity shift vrad + vsini
        """
        # 30 km/s == maximum barycentric velocity
        vrad_pad = 30.0 + 0.5 * np.clip(vsini, 0, None)  # km/s
        vbeg = vrad_pad + np.clip(vrad, 0, None)  # km/s
        vend = vrad_pad - np.clip(vrad, None, 0)  # km/s

        wbeg = wran[0] * (1 - vbeg / clight)
        wend = wran[1] * (1 + vend / clight)
        return wbeg, wend

    def new_wavelength_grid(self, wint):
        """ Generate new wavelength grid within bounds of wint"""
        # Determine step size for a new model wavelength scale, which must be uniform
        # to facilitate convolution with broadening kernels. The uniform step size
        # is the larger of:
        #
        # [1] smallest wavelength step in WINT_SEG, which has variable step size
        # [2] 10% the mean dispersion of WINT_SEG
        # [3] 0.05 km/s, which is 1% the width of solar line profiles

        wbeg, wend = wint[[0, -1]]
        wmid = 0.5 * (wend + wbeg)  # midpoint of segment
        wspan = wend - wbeg  # width of segment
        diff = wint[1:] - wint[:-1]
        jmin = np.argmin(diff)
        vstep1 = diff[jmin] / wint[jmin] * clight  # smallest step
        vstep2 = 0.1 * wspan / (len(wint) - 1) / wmid * clight  # 10% mean dispersion
        vstep3 = 0.05  # 0.05 km/s step
        vstep = max(vstep1, vstep2, vstep3)  # select the largest

        # Generate model wavelength scale X, with uniform wavelength step.
        nx = int(
            np.abs(np.log10(wend / wbeg)) / np.log10(1 + vstep / clight) + 1
        )  # number of wavelengths
        if nx % 2 == 0:
            nx += 1  # force nx to be odd

        # Resolution
        # IDL way
        # resol_out = 1 / ((wend / wbeg) ** (1 / (nx - 1)) - 1)
        # vstep = clight / resol_out
        # x_seg = wbeg * (1 + 1 / resol_out) ** np.arange(nx)

        # Python way (not identical, as IDL endpoint != wend)
        # difference approx 1e-7
        x_seg = np.geomspace(wbeg, wend, num=nx)
        resol_out = 1 / np.diff(np.log(x_seg[:2]))[0]
        vstep = clight / resol_out
        return x_seg, vstep

    @staticmethod
    def check_segments(sme, segments):
        if isinstance(segments, str) and segments == "all":
            segments = range(sme.nseg)
        else:
            segments = np.atleast_1d(segments)
            if np.any(segments < 0) or np.any(segments >= sme.nseg):
                raise IndexError("Segment(s) out of range")
        return segments

    def apply_radial_velocity_and_continuum(
        self, wave, wmod, smod, cmod, vrad, cscale, segments
    ):
        for il in segments:
            if vrad[il] is not None:
                rv_factor = np.sqrt((1 + vrad[il] / clight) / (1 - vrad[il] / clight))
                wmod[il] *= rv_factor
            smod[il] = safe_interpolation(wmod[il], smod[il], wave[il])
            if cmod is not None:
                cmod[il] = safe_interpolation(wmod[il], cmod[il], wave[il])

            if cscale[il] is not None and not np.all(cscale[il] == 0):
                x = wave[il] - wave[il][0]
                smod[il] *= np.polyval(cscale[il], x)
        return smod

    def synthesize_spectrum(
        self,
        sme,
        segments="all",
        passLineList=True,
        passAtmosphere=True,
        passNLTE=True,
        updateStructure=True,
        updateLineList=False,
        reuse_wavelength_grid=False,
        radial_velocity_mode="robust",
    ):
        """
        Calculate the synthetic spectrum based on the parameters passed in the SME structure
        The wavelength range of each segment is set in sme.wran
        The specific wavelength grid is given by sme.wave, or is generated on the fly if sme.wave is None

        Will try to fit radial velocity RV and continuum to observed spectrum, depending on vrad_flag and cscale_flag

        Other important fields:
        sme.iptype: instrument broadening type

        Parameters
        ----------
        sme : SME_Struct
            sme structure, with all necessary parameters for the calculation
        setLineList : bool, optional
            wether to pass the linelist to the c library (default: True)
        passAtmosphere : bool, optional
            wether to pass the atmosphere to the c library (default: True)
        passNLTE : bool, optional
            wether to pass NLTE departure coefficients to the c library (default: True)
        reuse_wavelength_grid : bool, optional
            wether to use sme.wint as the output grid of the function or create a new grid (default: False)

        Returns
        -------
        sme : SME_Struct
            same sme structure with synthetic spectrum in sme.smod
        """

        # Define constants
        n_segments = sme.nseg
        cscale_degree = sme.cscale_degree

        # fix impossible input
        if "spec" not in sme:
            sme.vrad_flag = "none"
            sme.cscale_flag = "none"
        else:
            for i in range(sme.nseg):
                sme.mask[i, ~np.isfinite(sme.spec[i])] = 0

        if "wint" not in sme:
            reuse_wavelength_grid = False
        if radial_velocity_mode != "robust" and (
            "cscale" not in sme or "vrad" not in sme
        ):
            radial_velocity_mode = "robust"

        segments = Synthesizer.check_segments(sme, segments)

        # Prepare arrays
        wran = sme.wran

        wint = [None for _ in range(n_segments)]
        sint = [None for _ in range(n_segments)]
        cint = [None for _ in range(n_segments)]
        vrad = np.zeros(n_segments)

        cscale = np.zeros((n_segments, cscale_degree + 1))
        cscale[:, -1] = 1
        wave = [None for _ in range(n_segments)]
        smod = [[] for _ in range(n_segments)]
        cmod = [[] for _ in range(n_segments)]
        wmod = [[] for _ in range(n_segments)]
        wind = np.zeros(n_segments + 1, dtype=int)

        # If wavelengths are already defined use those as output
        if "wave" in sme:
            wave = [w for w in sme.wave]
            wind = [0, *np.diff(sme.wind)]

        # Input Model data to C library
        self.dll.SetLibraryPath()
        if passLineList:
            self.dll.InputLineList(sme.linelist)
        if updateLineList:
            # TODO Currently Updates the whole linelist, could be improved to only change affected lines
            self.dll.UpdateLineList(
                sme.atomic, sme.species, np.arange(len(sme.linelist))
            )
        if passAtmosphere:
            sme = self.get_atmosphere(sme)
            self.dll.InputModel(sme.teff, sme.logg, sme.vmic, sme.atmo)
            self.dll.InputAbund(sme.abund)
            self.dll.Ionization(0)
            self.dll.SetVWscale(sme.gam6)
            self.dll.SetH2broad(sme.h2broad)
        if passNLTE:
            update_nlte_coefficients(sme, self.dll, self.lfs_nlte)

        # Loop over segments
        #   Input Wavelength range and Opacity
        #   Calculate spectral synthesis for each
        #   Interpolate onto geomspaced wavelength grid
        #   Apply instrumental and turbulence broadening

        # TODO Parallelization
        # This requires changes in the C code however, since SME uses global parameters
        # for the wavelength range (and opacities) which change within each segment
        for il in tqdm(segments, desc="Segment"):
            wmod[il], smod[il], cmod[il] = self.synthesize_segment(
                sme, il, reuse_wavelength_grid, il != segments[0]
            )

            if "wave" not in sme or len(sme.wave[il]) == 0:
                # trim padding
                wbeg, wend = sme.wran[il]
                itrim = (wmod[il] > wbeg) & (wmod[il] < wend)
                # Force endpoints == wavelength range
                wave[il] = np.concatenate(([wbeg], wmod[il][itrim], [wend]))

        # Fit continuum and radial velocity
        # And interpolate the flux onto the wavelength grid

        if radial_velocity_mode == "robust":
            cscale, cscale_unc, vrad, vrad_unc = match_rv_continuum(
                sme, segments, wmod, smod
            )
            logger.debug("Radial velocity: %s", str(vrad))
            logger.debug("Continuum coefficients: %s", str(cscale))
        elif radial_velocity_mode == "fast":
            cscale, vrad = sme.cscale, sme.vrad
        else:
            raise ValueError("Radial Velocity mode not understood")

        smod = self.apply_radial_velocity_and_continuum(
            wave, wmod, smod, cmod, vrad, cscale, segments
        )

        # Merge all segments
        # if sme already has a wavelength this should be the same
        if updateStructure:
            sme.wind = wind = np.cumsum(wind)
            sme.wint = wint

            if "wave" not in sme:
                npoints = sum([len(wave[s]) for s in segments])
                sme.wave = np.zeros(npoints)
            if "synth" not in sme:
                sme.synth = np.zeros_like(sme.wob)
            if "cont" not in sme:
                sme.cont = np.zeros_like(sme.wob)

            for s in segments:
                sme.wave[s] = wave[s]
                sme.synth[s] = smod[s]
                sme.cont[s] = cmod[s]

            if sme.cscale_flag not in ["fix", "none"]:
                for s in segments:
                    sme.cscale[s] = cscale[s]
                sme.cscale_unc = cscale_unc

            sme.vrad = np.asarray(vrad)
            sme.vrad_unc = np.asarray(vrad_unc)
            sme.nlte.flags = self.dll.GetNLTEflags()
            return sme
        else:
            wave = Iliffe_vector(values=wave)
            smod = Iliffe_vector(values=smod)
            return wave, smod

    def synthesize_segment(
        self, sme, segment, reuse_wavelength_grid, keep_line_opacity
    ):
        logger.debug("Segment %i", segment)

        # Input Wavelength range and Opacity
        vrad_seg = sme.vrad[segment] if sme.vrad[segment] is not None else 0
        wbeg, wend = self.get_wavelengthrange(sme.wran[segment], vrad_seg, sme.vsini)

        self.dll.InputWaveRange(wbeg, wend)
        self.dll.Opacity()

        # Reuse adaptive wavelength grid in the jacobians
        wint_seg = sme.wint[segment] if reuse_wavelength_grid else None
        # Only calculate line opacities in the first segment
        #   Calculate spectral synthesis for each
        _, wint, sint, cint = self.dll.Transf(
            sme.mu,
            sme.accrt,  # threshold line opacity / cont opacity
            sme.accwi,
            keep_lineop=keep_line_opacity,
            wave=wint_seg,
        )

        # Create new geomspaced wavelength grid, to be used for intermediary steps
        wgrid, vstep = self.new_wavelength_grid(wint)

        # Continuum
        # rtint = Radiative Transfer Integration
        cont_flux = integrate_flux(sme.mu, cint, 1, 0, 0)
        cont_flux = np.interp(wgrid, wint, cont_flux)

        # Broaden Spectrum
        y_integrated = np.empty((sme.nmu, len(wgrid)))
        for imu in range(sme.nmu):
            y_integrated[imu] = np.interp(wgrid, wint, sint[imu])

        # Turbulence broadening
        # Apply macroturbulent and rotational broadening while integrating intensities
        # over the stellar disk to produce flux spectrum Y.
        flux = integrate_flux(sme.mu, y_integrated, vstep, sme.vsini, sme.vmac)
        # instrument broadening
        if "iptype" in sme:
            ipres = sme.ipres if np.size(sme.ipres) == 1 else sme.ipres[segment]
            flux = broadening.apply_broadening(
                ipres, wgrid, flux, type=sme.iptype, sme=sme
            )

        # Divide calculated spectrum by continuum
        if sme.normalize_by_continuum:
            flux /= cont_flux

        return wgrid, flux, cont_flux


def synthesize_spectrum(sme, segments="all"):
    synthesizer = Synthesizer()
    return synthesizer.synthesize_spectrum(sme, segments)
