from scipy.io import readsav
import numpy as np

from .atmosphere import AtmosphereGrid


class SavFile(AtmosphereGrid):
    """ IDL savefile atmosphere grid """

    def __new__(cls, filename):
        data = readsav(filename)

        npoints = data["atmo_grid_maxdep"]
        ngrids = data["atmo_grid_natmo"]
        self = super(SavFile, cls).__new__(cls, ngrids, npoints)

        self.source = filename
        citation = [d.decode() for d in data["atmo_grid_intro"]]
        self.citation_info = "".join(citation)

        atmo_grid = data["atmo_grid"]

        if "RADIUS" in atmo_grid.dtype.names:
            self.geom = "SPH"
        else:
            self.geom = "PP"

        self.abund_format = "sme"

        # Scalar Parameters (one per atmosphere)
        self["teff"] = atmo_grid["teff"]
        self["logg"] = atmo_grid["logg"]
        self["monh"] = atmo_grid["monh"]
        self["vturb"] = atmo_grid["vturb"]
        self["lonh"] = atmo_grid["lonh"]
        # Vector Parameters (one array per atmosphere)
        self["rhox"] = np.stack(atmo_grid["rhox"])
        self["tau"] = np.stack(atmo_grid["tau"])
        self["temp"] = np.stack(atmo_grid["temp"])
        self["rho"] = np.stack(atmo_grid["rho"])
        self["xne"] = np.stack(atmo_grid["xne"])
        self["xna"] = np.stack(atmo_grid["xna"])
        self["abund"] = np.stack(atmo_grid["abund"])
        self["opflag"] = np.stack(atmo_grid["opflag"])
        return self
