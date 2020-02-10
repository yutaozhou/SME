import numpy as np

from scipy.constants import Boltzmann

from .atmosphere import Atmosphere
from ..abund import Abund
from ..sme import (
    CollectionFactory,
    asfloat,
    this,
    astype,
    absolute,
    asstr,
    lowercase,
    uppercase,
    oneof,
    array,
)


k_boltz = Boltzmann * 1e7


def read_record(buffer, offset=0):
    buffer = buffer[offset:]
    rl = np.frombuffer(buffer[:4], ">i4")[0]
    rl2 = np.frombuffer(buffer[4 + rl : 8 + rl], ">i4")[0]

    assert rl == rl2

    return 8 + rl, buffer[4 : 4 + rl]


@CollectionFactory
class MarcsFile(Atmosphere):
    def __init__(self, filename):
        super().__init__()
        self.source = filename
        self.method = "embedded"
        self.depth = "TAU"
        self.interp = "RHOX"
        self.lonh = 1.5
        self.loads(filename)

    def loads(self, filename):
        with open(filename, "rb") as file:
            content = file.read()

        # Record 1 = DUMMY,erad
        rl, data = read_record(content)
        dummy = np.frombuffer(data, ">f4")

        # Record 2, date and time
        content = content[rl:]
        rl, data = read_record(content)
        inord, dag, klock = np.frombuffer(data, ">i4,S10,S10")[0]
        dag = dag.decode().strip()
        klock = klock.decode().strip()

        # Record 3, Stellar Parameters
        content = content[rl:]
        rl, data = read_record(content)
        param, ip, name, itmax, nel = np.frombuffer(
            data[:72], "(7)>f4,(3)>i4,S24,>i4,>i4"
        )[0]
        abund = np.frombuffer(data[72 : 72 + 4 * nel], ">f4")
        # Scandium, Titanium, Vanadium, Manganese, Cobalt
        abund_other = np.frombuffer(data[72 + 4 * nel :], ">f4")
        teff, flux, g = param[:3]
        logg = np.log10(g)
        name = name.decode().strip()
        monh = float(name.partition("z")[2].partition("t")[0].partition("m")[0])
        vturb = float(name.partition("t")[2])

        abund = np.copy(abund)
        abund[1:] = np.log10(abund[1:]) + 12

        # Record 4, Radius
        content = content[rl:]
        rl, data = read_record(content)
        ntau, *param, radius = np.frombuffer(data, ">i4,>i4,>f4,>f4,>f4", count=1)[0]
        ncore, diflog, taum = param
        # Radius and RR are 1 for Plane Parallel
        rr = np.frombuffer(data, ">f4", count=ntau, offset=20)

        # Record 5 ?
        content = content[rl:]
        rl, data = read_record(content)
        ntau = np.frombuffer(data, ">i4", count=1)[0]  # also jtau
        tkorrm, fcorr = np.frombuffer(data, f"({ntau},)>f4", count=2, offset=4)
        ntpo = 0

        # Record 6, The Atnopshere grd data we want
        # The Next ntau records are similar
        # Rosseland Tau
        tau = np.zeros(ntau, ">f4")
        # Tau at 5000 Angstrom
        taus = np.zeros(ntau, ">f4")
        # Depth scale in cm
        z = np.zeros(ntau, ">f4")
        # Temperature profile
        temp = np.zeros(ntau, ">f4")
        # electron pressure
        pe = np.zeros(ntau, ">f4")
        # gas pressure (including pe)
        pg = np.zeros(ntau, ">f4")
        # radiation pressure
        prad = np.zeros(ntau, ">f4")
        # turbulence pressure
        pturb = np.zeros(ntau, ">f4")
        xkapr = np.zeros(ntau, ">f4")
        # Density
        rho = np.zeros(ntau, ">f4")
        # Mean molecular weight
        emu = np.zeros(ntau, ">f4")
        # Thermodynamic heat
        cp = np.zeros(ntau, ">f4")
        cv = np.zeros(ntau, ">f4")
        agrad = np.zeros(ntau, ">f4")
        q = np.zeros(ntau, ">f4")
        u = np.zeros(ntau, ">f4")
        v = np.zeros(ntau, ">f4")
        anconv = np.zeros(ntau, ">f4")
        hnic = np.zeros(ntau, ">f4")
        # Pressure related
        nmol = 30
        presmo = np.zeros((ntau, nmol), ">f4")
        ptio = np.zeros(ntau, ">f4")
        ptot = np.zeros(ntau, ">f4")

        for k in range(ntau):
            content = content[rl:]
            rl, data = read_record(content)
            kr, param, nmol = np.frombuffer(data, ">i4,(19,)>f4,>i4", count=1)[0]
            presmo[k], ptio[k] = np.frombuffer(
                data, f"({nmol})>f4,>f4", count=1, offset=84
            )[0]

            (tau[k], taus[k], z[k], temp[k], pe[k], pg[k], prad[k], pturb[k],) = param[
                :8
            ]
            xkapr[k], rho[k], emu[k], cp[k], cv[k], agrad[k], q[k], u[k] = param[8:16]
            v[k], anconv[k], hnic[k] = param[16:19]

            tauk = np.log10(tau[k]) + 10.01
            ktau = int(tauk)

            if abs(tauk - ktau) > 0.02:
                continue
            elif ktau == 10:
                # "First" k is identifiend by log10(tau) == 10
                k0 = k
            ntpo += 1

        z0 = z[k0]
        for i in range(ntau):
            z[i] = z[i] - z[0]
            i1 = min(i + 1, ntau - 1)
            ptot[i] = pg[i] + prad[i] + 0.5 * (pturb[i] + pturb[i1])

        xna = (pg - pe) / (k_boltz * temp)
        xne = pe / (k_boltz * temp)

        # Record 7
        content = content[rl:]
        rl, data = read_record(content)

        nj, nlp = np.frombuffer(data, f"({nel})>i4,>i4", count=1)[0]
        xlr, nprov, nprova, nprovs = np.frombuffer(
            data, f"({nlp})>i4,>i4,>i4,>i4", count=1, offset=4 + nel * 4
        )[0]
        abname = np.frombuffer(data, "(2,)S8", count=nprov, offset=16 + (nel + nlp) * 4)
        abname = np.char.decode(abname)
        abname = np.char.strip(abname)
        abname, source = abname[:, 0], abname[:, 1]

        # Record 8
        njp = np.max(nj)
        iel = np.zeros(nel, ">i4")
        anjon = np.zeros((nel, njp), ">f4")
        part = np.zeros((nel, njp), ">f4")
        prov = np.zeros((nprov, nlp), ">f4")
        abska = np.zeros(nlp, ">f4")
        sprida = np.zeros(nlp, ">f4")

        for ktau in range(ntpo):
            for ie in range(nel):
                content = content[rl:]
                rl, data = read_record(content)

                njp = nj[ie]
                kr, param, anjon[ie, :njp], part[ie, :njp] = np.frombuffer(
                    data, f">i4,(5)>f4,({njp})>f4,({njp})>f4"
                )[0]
                taui, ti, pei, iel[ie], abund[ie] = param

            for klam in range(nlp):
                content = content[rl:]
                rl, data = read_record(content)

                kr, tauil, prov[:, klam], abska[klam], sprida[klam] = np.frombuffer(
                    data, f">i4,>f4,({nprov})>f4,>f4,>f4"
                )[0]

        # Record 9
        content = content[rl:]
        rl, data = read_record(content)

        nlb = np.frombuffer(data, ">i4", count=1)[0]
        temp2 = np.frombuffer(data, f"({nlb},2)>f4,({nlb})>f4", count=1, offset=4)[0]
        xlb = temp2[0][:, 0]
        fluxme = temp2[0][:, 1]
        w = temp2[1]

        fluxme = fluxme * np.pi
        dluminosity = np.sum(fluxme * w)
        dluminosity = dluminosity * 4.0 * np.pi * radius ** 2 / 3.82e33

        # Record 10
        content = content[rl:]
        rl, data = read_record(content)

        fluxme_cont = np.frombuffer(data, ">f4", count=nlb)
        fluxme_cont = fluxme_cont * np.pi

        # Record 11
        content = content[rl:]
        rl, data = read_record(content)

        abundance = np.full(99, -99)
        abundance[:92] = np.frombuffer(data, ">f4", count=92)

        # compute RHOX, the integrated column mass from the surface to here
        # units g/cm2, modified April 2018 from previously ptot/gg
        rhox = np.zeros(ntau, ">f4")
        geom = "SPH" if radius > 1 else "PP"
        if geom == "PP":
            rhox[0] = ptot[0] / g
            rhox[1:] = (z[1:] - z[:-1]) * (rho[:-1] + rho[1:]) / 2
            rhox = np.cumsum(rhox)
        elif geom == "SPH":
            # With Spherical correction
            rhox[0] = ptot[0] / (g * (radius / rr[0]) ** 2)
            # add what is in a paper cup of 1 cm2 bottom area between this and
            # the level above plus the mass above that corrected for the top area
            for i in range(1, ntau):
                rhox[i] = rhox[i - 1] * (rr[i - 1] / rr[i]) ** 2 + (
                    rho[i] + rho[i - 1]
                ) / 6 * (rr[i - 1] * (rr[i - 1] / rr[i]) ** 2 - rr[i])

        self.teff = teff
        self.logg = logg
        self.vturb = vturb
        self.radius = radius
        self.height = rr[0]  # ???
        self.geom = geom
        self.opflag = np.ones(20, ">i4")
        self.abund = Abund(monh=monh, pattern=abundance, type="H=12")

        self.tau = taus
        self.rho = rho
        self.rhox = rhox
        self.temp = temp
        self.xna = xna
        self.xne = xne

    @staticmethod
    def load(filename):
        return MarcsFile(filename)
