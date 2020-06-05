"""
NLTE module of SME
reads and interpolates departure coefficients from library files
"""

import itertools
import logging
import warnings

import numpy as np
from scipy import interpolate

from .abund import Abund

logger = logging.getLogger(__name__)


class DirectAccessFile:
    """
    This function reads a single record from binary file that has the following
    structure:
    Version string              - 64 byte long
    # of directory bocks        - short int
    directory block length      - short int
    # of used directory blocks  - short int
    1st directory block
    key      - string of up to 256 characters padded with ' '
    datatype - 32-bit int 23-element array returned by SIZE
    pointer  - 64-bit int pointer to the beginning of the record

    2nd directory block
    ...
    last directory block
    1st data record
    ...
    last data record
    """

    def __init__(self, filename):
        key, pointer, dtype, shape, version = DirectAccessFile.read_header(filename)
        self.file = filename
        self.version = version
        self.key = key
        self.shape = shape
        self.pointer = pointer
        self.dtype = dtype

    def __getitem__(self, key):
        # Access data via brackets
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found")
        return value

    def get(self, key, alt=None):
        """ get field from file """
        idx = np.where(self.key == key)[0]

        if idx.size == 0:
            return alt
        else:
            idx = idx[0]

        return np.memmap(
            self.file,
            mode="r",
            offset=self.pointer[idx],
            dtype=self.dtype[idx],
            shape=self.shape[idx][::-1],
        )

    @staticmethod
    def idl_typecode(i):
        """
        relevant IDL typecodes and corresponding Numpy Codes
        Most specify a byte size, but not all
        """
        typecode = {
            0: "V",
            1: "B",
            2: "i2",
            3: "i4",
            4: "f4",
            5: "f8",
            6: "c4",
            7: "S",
            8: "O",
            9: "c8",
            10: "i8",
            11: "O",
            12: "u2",
            13: "u4",
            14: "i8",
            15: "u8",
        }
        return typecode[i]

    @staticmethod
    def get_typecode(dt):
        """
        relevant IDL typecodes and corresponding Numpy Codes
        Most specify a byte size, but not all
        """
        typecode = {
            "V": 0,
            "B": 1,
            "U": 1,
            "i2": 2,
            "i4": 3,
            "f4": 4,
            "f8": 5,
            "c4": 6,
            "S": 7,
            "O": 8,
            "c8": 9,
            "i8": 10,
            "O": 11,
            "u2": 12,
            "u4": 13,
            "i8": 14,
            "u8": 15,
        }
        if isinstance(dt, np.dtype):
            dt = dt.str
        if len(dt) > 2:
            dt = dt[1:]

        return typecode[dt]

    @staticmethod
    def get_dtypes(major, minor):
        if major == "1" and minor == "00":
            header_dtype = np.dtype(
                [("nblocks", "<u2"), ("dir_length", "<u2"), ("ndir", "<u2"),]
            )
            dir_dtype = np.dtype(
                [("key", "S256"), ("size", "<i4", 23), ("pointer", "<i8")]
            )
        elif major == "1" and minor == "10":
            header_dtype = np.dtype(
                [("nblocks", "<u8"), ("dir_length", "<u2"), ("ndir", "<u8"),]
            )
            dir_dtype = np.dtype(
                [("key", "S256"), ("size", "<i4", 23), ("pointer", "<i8")]
            )
        else:
            raise ValueError("DirectAccess File Version '{version}' not understood.")
        return header_dtype, dir_dtype

    @staticmethod
    def read_header(fname):
        """ parse Header data """
        with open(fname, "rb") as file:
            version_dtype = "S64"
            version = np.fromfile(file, version_dtype, count=1)
            version = version[0].decode()
            major, minor = version[26], version[28:30]
            version = (major, minor)
            header_dtype, dir_dtype = DirectAccessFile.get_dtypes(major, minor)

            header = np.fromfile(file, header_dtype, count=1)
            ndir = int(header["ndir"][0])

            src = np.fromfile(file, dir_dtype, count=1)
            # TODO: Temporary fix
            ndir -= 1
            file.read(64 * 2 - 16)
            # End fix
            directory = np.fromfile(file, dir_dtype, count=ndir)

        # Decode bytes to strings
        key = directory["key"]
        key = np.char.strip(np.char.decode(key))
        # Get relevant info from size parameter
        # ndim, n1, n2, ..., typecode, size
        dtype = np.array(
            [DirectAccessFile.idl_typecode(d[1 + d[0]]) for d in directory["size"]],
            dtype="U5",
        )
        shape = np.empty(ndir, dtype=object)
        shape[:] = [tuple(d[1 : d[0] + 1]) for d in directory["size"]]
        # Pointer to data arrays
        pointer = directory["pointer"]

        # Bytes (which represent strings) get special treatment to get the dimensions right
        # And to properly convert them to strings
        # Also Null terminator is important to remember
        idx = dtype == "B"
        dtype[idx] = [f"S{s[0]}" for s in shape[idx]]
        shape2 = np.empty(np.count_nonzero(idx), dtype=object)
        shape2[:] = [tuple(s[1:]) for s in shape[idx]]
        shape[idx] = shape2

        return key, pointer, dtype, shape, version

    @staticmethod
    def write(fname, **kwargs):
        major, minor = "1", "10"
        ndir = len(kwargs)
        version = f"DirectAccess file Version {major}.{minor} 2011-03-24"
        version = np.asarray([version], dtype="S64")
        version_length = version.itemsize

        header_dtype, dir_dtype = DirectAccessFile.get_dtypes(major, minor)
        header_length = header_dtype.itemsize
        dir_length = dir_dtype.itemsize

        header = np.zeros(1, header_dtype)
        header[0]["nblocks"] = len(kwargs)
        header[0]["dir_length"] = dir_length
        header[0]["ndir"] = ndir

        directory = np.zeros(ndir, dir_dtype)
        pointer = version_length + header_length + ndir * dir_length

        for i, (key, value) in enumerate(kwargs.items()):
            value = np.asarray(value)

            directory[i]["key"] = key
            shape = directory[i]["size"]
            if np.issubdtype(value.dtype, np.dtype("U")) or np.issubdtype(
                value.dtype, np.dtype("S")
            ):
                value = value.astype("S")
                shape[0] = value.ndim + 1
                shape[1 : 2 + value.ndim] = value.itemsize, *value.shape
                shape[2 + value.ndim] = 1
                shape[3 + value.ndim] = value.size * value.itemsize
            else:
                shape[0] = value.ndim
                shape[1 : 1 + value.ndim] = value.shape
                shape[1 + value.ndim] = DirectAccessFile.get_typecode(value.dtype)
                shape[2 + value.ndim] = value.size

            directory[i]["pointer"] = pointer
            pointer += value.nbytes
            kwargs[key] = value

        with open(fname, "wb") as file:
            version.tofile(file)
            header.tofile(file)
            directory.tofile(file)

            for i, value in enumerate(kwargs.values()):
                value.tofile(file)


class Grid:
    """NLTE Grid class that handles all NLTE data reading and interpolation
    """

    def __init__(self, sme, elem, lfs_nlte, selection="energy"):
        #:str: Element of the NLTE grid
        self.elem = elem
        #:LineList: Whole LineList that was passed to the C library
        self.linelist = sme.linelist
        #:array(str): Elemental Species Names for the linelist
        self.species = sme.linelist.species
        #:str: complete filename of the NLTE grid data file
        self.fname = lfs_nlte.get(sme.nlte.grids[elem])
        depth_name = str.lower(sme.atmo.interp)
        #:array(float): depth points of the atmosphere that was passed to the C library (in log10 scale)
        self.target_depth = sme.atmo[depth_name]
        self.target_depth = np.log10(self.target_depth)
        #:{"levels", "energy"}: Selection algorithm to match lines in grid with linelist
        self.selection = selection

        #:DirectAccessFile: The NLTE data file
        self.directory = DirectAccessFile(self.fname)
        self.version = self.directory.version
        # The possible labels
        self._teff = self.directory["teff"]
        self._grav = self.directory["grav"]
        self._feh = self.directory["feh"]
        self._xfe = self.directory["abund"]
        # The position of the models in the datafile
        self._keys = self.directory["models"].astype("U")
        self._depth = self.directory[depth_name]

        self._grid = None
        self._points = None

        #:list(int): number of points in the grid to cache for each parameter, order; abund, teff, logg, monh
        self.subgrid_size = sme.nlte.subgrid_size
        #:float: Solar Abundance of the element
        self.solar = Abund.solar()[self.elem]

        #:dict: upper and lower parameters covered by the grid
        self.limits = {}
        #:array: NLTE data array
        self.bgrid = None
        #:array: Depth points of the NLTE data
        self.depth = None

        #:array: Indices of the lines in the NLTE data
        self.linerefs = None
        #:array: Indices of the lines in the LineList
        self.lineindices = None
        #:array: Indices of the lines in the bgrid
        self.iused = None

        #:str: citations in bibtex format, if known
        self.citation_info = ""

        self.line_match_mode = "level"

        conf = self.directory["conf"].astype("U")
        term = self.directory["term"].astype("U")
        species = self.directory["spec"].astype("U")
        rotnum = self.directory["J"]  # rotational number of the atomic state
        if self.version[0] == "1" and self.version[1] == "10":
            energies = self.directory["energy"]  # energy in eV
            self.citation_info = self.directory["citation"]
        else:
            self.citation_info = None
            if self.selection != "levels":
                logger.warning(
                    "NLTE grid file version %s only supports level selection, not %s",
                    self.version,
                    self.selection,
                )
                self.selection = "levels"

        if self.selection == "levels":
            self.lineindices, self.linerefs, self.iused = self.select_levels(
                conf, term, species, rotnum
            )
        elif self.selection == "energy":
            self.lineindices, self.linerefs, self.iused = self.select_energies(
                conf, term, species, rotnum, energies
            )

    def get(self, abund, teff, logg, monh):
        # TODO: NLTE grid should define which solar value has been used
        rabund = abund - self.solar

        if len(self.limits) == 0 or not (
            (self.limits["xfe"][0] <= rabund <= self.limits["xfe"][-1])
            and (self.limits["teff"][0] <= teff <= self.limits["teff"][-1])
            and (self.limits["grav"][0] <= logg <= self.limits["grav"][-1])
            and (self.limits["feh"][0] <= monh <= self.limits["feh"][-1])
        ):
            _ = self.read_grid(rabund, teff, logg, monh)

        return self.interpolate(rabund, teff, logg, monh)

    def read_grid(self, rabund, teff, logg, monh):
        """ Read the NLTE coefficients from the nlte_grid files for the given element
        The class will cache subgrid_size points around the target values as well

        Parameters
        ----------
        rabund : float
            relative (to solar) abundance of the element
        teff : float
            temperature in Kelvin
        logg : float
            surface gravity in log(cgs)
        monh : float
            Metallicity in H=12

        Returns
        -------
        nlte_grid : dict
            collection of nlte coefficient data (memmapped)
        linerefs : array (nlines,)
            linelevel descriptions (Energy level terms)
        lineindices: array (nlines,)
            indices of the used lines in the linelist
        """
        # find the n nearest parameter values in the grid (n == subgrid_size)
        x = np.argsort(np.abs(rabund - self._xfe))[: self.subgrid_size[0]]
        t = np.argsort(np.abs(teff - self._teff))[: self.subgrid_size[1]]
        g = np.argsort(np.abs(logg - self._grav))[: self.subgrid_size[2]]
        f = np.argsort(np.abs(monh - self._feh))[: self.subgrid_size[3]]

        x = x[np.argsort(self._xfe[x])]
        t = t[np.argsort(self._teff[t])]
        g = g[np.argsort(self._grav[g])]
        f = f[np.argsort(self._feh[f])]

        # Read the models with those parameters, and store depth and level
        # Create storage array
        ndepths, nlevel = self.directory[self._keys[0, 0, 0, 0]].shape
        nabund = len(x)
        nteff = len(t)
        ngrav = len(g)
        nfeh = len(f)

        self.bgrid = np.zeros((ndepths, nlevel, nabund, nteff, ngrav, nfeh))

        for i, j, k, l in np.ndindex(nabund, nteff, ngrav, nfeh):
            model = self._keys[f[l], g[k], t[j], x[i]]
            try:
                self.bgrid[:, :, i, j, k, l] = self.directory[model]
            except KeyError:
                warnings.warn(
                    f"Missing Model for element {self.elem}: T={self._teff[t[j]]}, logg={self._grav[g[k]]}, feh={self._feh[f[l]]}, abund={self._xfe[x[i]]:.2f}"
                )
        mask = np.zeros(self._depth.shape[:-1], bool)
        for i, j, k in itertools.product(f, g, t):
            mask[i, j, k] = True
        self.depth = self._depth[mask, :]
        self.depth.shape = nfeh, ngrav, nteff, -1

        # Reduce the stored data to only relevant energy levels
        # Remap the previous indices into a collapsed sequence
        # level_labels = level_labels[iused]
        self.bgrid = self.bgrid[self.iused, ...]

        self._points = (self._xfe[x], self._teff[t], self._grav[g], self._feh[f])
        self.limits = {
            "teff": self._points[1][[0, -1]],
            "grav": self._points[2][[0, -1]],
            "feh": self._points[3][[0, -1]],
            "xfe": self._points[0][[0, -1]],
        }

        # Interpolate the depth scale to the target depth, this is unstructured data
        # i.e. each combination of parameters has a different depth scale (given in depth)
        ndepths, _, *nparam = self.bgrid.shape
        ntarget = len(self.target_depth)

        self._grid = np.empty((*nparam, ntarget, ndepths), float)
        for l, x, t, g, f in np.ndindex(ndepths, *nparam):
            xp = self.depth[f, g, t, :]
            yp = self.bgrid[l, :, x, t, g, f]

            xp = np.log10(xp)
            self._grid[x, t, g, f, :, l] = interpolate.interp1d(
                xp, yp, bounds_error=False, fill_value=(yp[0], yp[-1]), kind="cubic"
            )(self.target_depth)

        return self.bgrid

    def select_levels(self, conf, term, species, rotnum):
        """
        Match our NLTE terms to transitions in the vald3-format linelist.

        Level descriptions in the vald3 long format look like this:
        'LS                 2p6.3s               2S'
        'LS                 2p6.3p               2P*'
        These are stored in line_term_low and line_term_upp.
        The array line_extra has dimensions [3 x nlines]. It stores J_low, E_up, J_up
        The sme.atomic array stores:
        0) atomic number, 1) ionization state, 2) wavelength (in A),
        3) excitation energy of lower level (in eV), 4) log(gf), 5) radiative,
        6) Stark, 7) and van der Waals damping parameters

        Parameters
        ----------
        conf : array of shape (nl,)
            electronic configuration (for identification), e.g., 2p6.5s
        term : array of shape (nl,)
            term designations (for identification), e.g., a5F
        species : array of shape (nl,)
            Element and ion for each atomic level (for identification), e.g. Na 1.
        rotnum : array of shape (nl,)
            rotational number J of atomic state (for identification).

        Returns
        -------
        lineindices : array of shape (nl,)
            Indices of the used lines in the linelist
        linerefs : array of shape (2, nlines,)
            Cross references for the lower and upper level in each transition,
            to their indices in the list of atomic levels.
            Missing levels use indices of -1.
        iused : array of shape (nl,)
            Indices used in the subgrid
        """

        lineindices = np.asarray(self.species, "U")
        lineindices = np.char.startswith(lineindices, self.elem)
        if not np.any(lineindices):
            warnings.warn(f"No NLTE transitions for {self.elem} found")
            return None, None, None

        sme_species = self.species[lineindices]

        # Extract data from linelist
        low = self.linelist["term_lower"][lineindices]
        upp = self.linelist["term_upper"][lineindices]
        # Remove quotation marks (if any are there)
        # parts_low = [s.replace("'", "") for s in parts_low]
        # parts_upp = [s.replace("'", "") for s in parts_upp]
        # Get only the relevant part
        parts_low = np.char.partition(low, " ")[:, (0, 2)]
        parts_upp = np.char.partition(upp, " ")[:, (0, 2)]

        # Transform into term symbol J (2*S+1) ?
        extra = self.linelist.extra[lineindices]
        extra = extra[:, [0, 2]] * 2 + 1
        extra = np.rint(extra).astype("i8")

        # Transform into term symbol J (2*S+1) ?
        rotnum = np.rint(2 * rotnum + 1).astype(int)

        # Create record arrays for each set of labels
        dtype = [
            ("species", sme_species.dtype),
            ("configuration", parts_upp.dtype),
            ("term", parts_upp.dtype),
            ("J", extra.dtype),
        ]
        level_labels = np.rec.fromarrays((species, conf, term, rotnum), dtype=dtype)
        line_label_low = np.rec.fromarrays(
            (sme_species, parts_low[:, 0], parts_low[:, 1], extra[:, 0]), dtype=dtype
        )
        line_label_upp = np.rec.fromarrays(
            (sme_species, parts_upp[:, 0], parts_upp[:, 1], extra[:, 1]), dtype=dtype
        )

        # Prepare arrays
        nlines = parts_low.shape[0]
        linerefs = np.full((nlines, 2), -1)
        iused = np.zeros(len(species), dtype=bool)

        # Loop through the NLTE levels
        # and match line levels
        for i, level in enumerate(level_labels):
            idx_l = line_label_low == level
            linerefs[idx_l, 0] = i
            iused[i] = iused[i] or np.any(idx_l)

            idx_u = line_label_upp == level
            linerefs[idx_u, 1] = i
            iused[i] = iused[i] or np.any(idx_u)

        lineindices = np.where(lineindices)[0]

        # Remap the linelevel references
        for j, i in enumerate(np.where(iused)[0]):
            linerefs[linerefs == i] = j

        return lineindices, linerefs, iused

    def select_energies(self, conf, term, species, rotnum, energies):
        lineindices = np.asarray(self.species, "U")
        lineindices = np.char.startswith(lineindices, self.elem)
        if not np.any(lineindices):
            warnings.warn(f"No NLTE transitions for {self.elem} found")
            return None, None, None
        sme_species = self.species[lineindices]

        # Extract data from linelist
        term_low = self.linelist["term_lower"][lineindices]
        term_upp = self.linelist["term_upper"][lineindices]
        # Split the string into configuration and term
        term_low = np.char.partition(term_low, " ")
        term_upp = np.char.partition(term_upp, " ")
        # Remove whitespaces
        term_low = np.char.replace(term_low, " ", "")
        term_upp = np.char.replace(term_upp, " ", "")
        # Get only the relevant part
        conf_low, term_low = term_low[:, 0], term_low[:, 2]
        conf_upp, term_upp = term_upp[:, 0], term_upp[:, 2]

        # Energy levels in the linelist in eV
        energy_low = self.linelist["excit"][lineindices]
        energy_upp = self.linelist["e_upp"][lineindices]

        # Transform into term symbol J (2*S+1) ?
        extra = self.linelist.extra[lineindices]
        extra = extra[:, [0, 2]] * 2 + 1
        extra = np.rint(extra).astype(int)
        j_low, j_upp = extra[:, 0], extra[:, 1]

        # Transform into term symbol J (2*S+1) ?
        rotnum = np.rint(2 * rotnum + 1).astype(int)

        dtype = [
            ("species", sme_species.dtype),
            ("configuration", term_low.dtype),
            ("term", term_low.dtype),
            ("J", extra.dtype),
            ("energy", energy_low.dtype),
        ]

        level_labels = np.rec.fromarrays(
            (species, conf, term, rotnum, energies), dtype=dtype
        )
        line_label_low = np.rec.fromarrays(
            (sme_species, conf_low, term_low, j_low, energy_low), dtype=dtype
        )
        line_label_upp = np.rec.fromarrays(
            (sme_species, conf_upp, term_upp, j_upp, energy_upp), dtype=dtype
        )

        nlines = term_low.shape[0]  # np.count_nonzero(lineindices)
        linerefs = np.full((nlines, 2), -1)
        iused = np.zeros(len(species), dtype=bool)

        idx_map = np.arange(nlines)
        # Maximum energy in the grid
        max_energy = np.max(energies)
        # Maximum seperation between energy levels
        energy_diff_limit = np.max(np.abs(np.diff(energies)))

        def match(label, level):
            # Try to match the label and the level
            # Using various metrics in decreasing order of confidence
            idx_species = label.species == level.species
            idx_conf = label.configuration == level.configuration
            idx_term = label.term == level.term
            idx_j = label.J == level.J
            # 1. Try to match conf/term/spec/J as usual
            idx = idx_species & idx_conf & idx_term & idx_j
            if np.any(idx):
                return idx
            # 2. If it fails, try to match conf/term/spec, but ignore J
            idx = idx_species & idx_conf & idx_term
            if np.any(idx):
                return idx
            # 3. Try to match H
            # If spec is 'H 1', then try to match conf with conf,
            # *or* try to match term with term, *or* try to match conf with term,
            # *or* try to match term with conf
            if level.species == "H 1":
                # TODO
                idx = (
                    idx_conf
                    | idx_term
                    | (label.term == level.configuration)
                    | (label.configuration == level.term)
                )
                if np.any(idx):
                    return idx
            # 4. Try to match energies (including H, if step 3 failed)
            # Find the level in the nlte grid with the same spec,
            # and the closest energy; *provided* that the desired energy does
            # not exceed the highest energy out of all the levels in the nlte grid with this spec
            idx = idx_species
            if np.any(idx):
                if level.energy > max_energy or level.energy < 0:
                    # We exceed the maximum energy of the grid, ignore NLTE
                    return []
                diff = np.abs(label[idx].energy - level.energy)
                idx2 = np.argmin(diff)
                mindiff = diff[idx2]
                # difference needs to be smaller than some limit?
                if mindiff < energy_diff_limit:
                    return idx_map[idx][idx2]
                else:
                    return []
            # 5. If everything fails return nothing
            return []

        # Loop through the NLTE levels
        # and match line levels
        for i, level in enumerate(level_labels):
            idx_l = match(line_label_low, level)
            linerefs[idx_l, 0] = i
            iused[i] |= np.any(idx_l)

            idx_u = match(line_label_upp, level)
            linerefs[idx_u, 1] = i
            iused[i] |= np.any(idx_u)

        return lineindices, linerefs, iused

    def interpolate(self, rabund, teff, logg, monh):
        """
        interpolate nlte coefficients on the model grid

        Parameters
        ----------
        rabund : float
            relative (to solar) abundance of the element
        teff : float
            temperature in Kelvin
        logg : float
            surface gravity in log(cgs)
        monh : float
            Metallicity in H=12

        Returns
        -------
        subgrid : array (ndepth, nlines)
            interpolated grid values
        """

        assert self._grid is not None
        assert self._points is not None

        # Interpolate on the grid
        # self._points and self._grid are interpolated when reading the data in read_grid
        target = (rabund, teff, logg, monh)
        points = self._points
        grid = self._grid

        # Check if we need to extrapolate
        if any([t < min(p) or t > max(p) for t, p in zip(target, points)]):
            logger.warning(
                f"Extrapolate on the NLTE grid. Requested values of {target} on grid {points}"
            )

        # Some grids have only one value in that direction
        # Usually in abundance. Then we need to remove that dimension
        # to avoid nan output
        mask = [len(p) > 1 for p in points]
        if not all(mask):
            points = [p for m, p in zip(mask, points) if m]
            target = [t for m, t in zip(mask, target) if m]
            idx = [slice(None, None) if m else 0 for m in mask]
            grid = grid[idx]

        # TODO: Interpolate with splines
        # Possibly in order of importance, since scipy doesn't have spline interpolation on a grid
        subgrid = interpolate.interpn(
            points, grid, target, method="linear", bounds_error=False, fill_value=None,
        )

        return subgrid[0]


def nlte(sme, dll, elem, lfs_nlte):
    """ Read and interpolate the NLTE grid for the current element and parameters """
    if sme.nlte.grids[elem] is None:
        raise ValueError(f"Element {elem} has not been prepared for NLTE")

    # TODO: The grids are cached in the DLL object
    # Its probably better to store them in a seperate object
    if elem in sme.nlte._nlte_grids.keys():
        grid = sme.nlte._nlte_grids[elem]
    else:
        grid = Grid(sme, elem, lfs_nlte)
        sme.nlte._nlte_grids[elem] = grid

    subgrid = grid.get(sme.abund[elem], sme.teff, sme.logg, sme.monh)

    return subgrid, grid.linerefs, grid.lineindices


# TODO should this be in sme_synth instead ?
def update_nlte_coefficients(sme, dll, lfs_nlte):
    """ pass departure coefficients to C library """

    # Only print "Running in NLTE" message on the first run each time
    self = update_nlte_coefficients
    if not hasattr(self, "first"):
        setattr(self, "first", True)

    if (
        not "nlte" in sme
        or "elements" not in sme.nlte
        or "grids" not in sme.nlte
        or np.all(sme.nlte.grids == "")
        or np.size(sme.nlte.elements) == 0
    ):
        # No NLTE to do
        if self.first:
            self.first = False
            logger.info("Running in LTE")
        return sme
    if sme.linelist.lineformat == "short":
        if self.first:
            self.first = False
            warnings.warn(
                "NLTE line formation was requested, but VALD3 long-format linedata\n"
                "are required in order to relate line terms to NLTE level corrections!\n"
                "Line formation will proceed under LTE."
            )
        return sme

    if self.first:
        self.first = False
        logger.info("Running in NLTE")

    # Reset the departure coefficient every time, just to be sure
    # It would be more efficient to just Update the values, but this doesn't take long
    dll.ResetNLTE()

    elements = sme.nlte.elements

    if self.first:
        self.first = False
        logger.info("Running in NLTE: %s", ", ".join(elements))

    # Call each element to update and return its set of departure coefficients
    for elem in elements:
        # Call function to retrieve interpolated NLTE departure coefficients
        bmat, linerefs, lineindices = nlte(sme, dll, elem, lfs_nlte)

        if bmat is None or len(linerefs) < 2:
            # no data were returned. Don't bother?
            pass
        else:
            # Put corrections into the nlte_b matrix, don't cache the data
            for lr, li in zip(linerefs, lineindices):
                # loop through the list of relevant _lines_, substitute both their levels into the main b matrix
                # Make sure both levels have corrections available
                if lr[0] != -1 and lr[1] != -1:
                    dll.InputNLTE(bmat[:, lr].T, li)

    # flags = sme_synth.GetNLTEflags(sme.linelist)

    return sme
