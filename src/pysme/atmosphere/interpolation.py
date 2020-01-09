""" Handles reading and interpolation of atmopshere (grid) data """
import itertools
import logging
import os
from os.path import dirname, join

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from scipy.io import readsav

from .atmosphere import Atmosphere as Atmo, AtmosphereError
from .savfile import SavFile

tf.config.optimizer.set_jit(True)


logger = logging.getLogger(__name__)


class AtmosphereError(RuntimeError):
    """ Something went wrong with the atmosphere interpolation """


class sav_file(np.recarray):
    """ IDL savefile atmosphere grid """

    def __new__(cls, filename, lfs_atmo):
        cls.lfs_atmo = lfs_atmo
        path = lfs_atmo.get(filename)
        krz2 = readsav(path)
        atmo_grid = krz2["atmo_grid"]
        atmo_grid_maxdep = krz2["atmo_grid_maxdep"]
        atmo_grid_natmo = krz2["atmo_grid_natmo"]
        atmo_grid_vers = krz2["atmo_grid_vers"]
        atmo_grid_file = filename
        atmo_grid_intro = krz2["atmo_grid_intro"]

        # Keep values around for next run
        data = atmo_grid.view(cls)
        data.maxdep = atmo_grid_maxdep
        data.natmo = atmo_grid_natmo
        data.vers = atmo_grid_vers
        data.file = atmo_grid_file
        data.intro = atmo_grid_intro

        return data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.maxdep = getattr(self, "maxdep", None)
        self.natmo = getattr(self, "natmo", None)
        self.vers = getattr(self, "vers", None)
        self.file = getattr(self, "file", None)
        self.intro = getattr(self, "intro", None)

    def load(self, filename, reload=False):
        """ load a new file if necessary """
        changed = filename != self.file
        if reload or changed:
            new = sav_file(filename, self.lfs_atmo)
        else:
            new = self
        return new


class krz_file(Atmo):
    """ Read .krz atmosphere files """

    def __init__(self, filename):
        super().__init__()
        self.source = filename
        self.method = "embedded"
        self.load(filename)

    def load(self, filename):
        """
        Load data from disk

        Parameters
        ----------
        filename : str
            name of the file to load
        """
        # TODO: this only works for some krz files
        # 1..2 lines header
        # 3 line opacity
        # 4..13 elemntal abundances
        # 14.. Table data for each layer
        #    Rhox Temp XNE XNA RHO

        with open(filename, "r") as file:
            header1 = file.readline()
            header2 = file.readline()
            opacity = file.readline()
            abund = [file.readline() for _ in range(10)]
            table = file.readlines()

            # Parse header
            # vturb
        i = header1.find("VTURB")
        self.vturb = float(header1[i + 5 : i + 9])
        # L/H, metalicity
        i = header1.find("L/H")
        self.lonh = float(header1[i + 3 :])

        k = len("T EFF=")
        i = header2.find("T EFF=")
        j = header2.find("GRAV=", i + k)
        self.teff = float(header2[i + k : j])

        i = j
        k = len("GRAV=")
        j = header2.find("MODEL TYPE=", i + k)
        self.logg = float(header2[i + k : j])

        i, k = j, len("MODEL TYPE=")
        j = header2.find("WLSTD=", i + k)
        model_type_key = {0: "rhox", 1: "tau", 3: "sph"}
        self.model_type = int(header2[i + k : j])
        self.depth = model_type_key[self.model_type]
        self.geom = "pp"

        i = j
        k = len("WLSTD=")
        self.wlstd = float(header2[i + k :])

        # parse opacity
        i = opacity.find("-")
        opacity = opacity[:i].split()
        self.opflag = np.array([int(k) for k in opacity])

        # parse abundance
        pattern = np.genfromtxt(abund).flatten()[:-1]
        pattern[1] = 10 ** pattern[1]
        self.abund = Abund(monh=0, pattern=pattern, type="sme")

        # parse table
        self.table = np.genfromtxt(table, delimiter=",", usecols=(0, 1, 2, 3, 4))
        self.rhox = self.table[:, 0]
        self.temp = self.table[:, 1]
        self.xne = self.table[:, 2]
        self.xna = self.table[:, 3]
        self.rho = self.table[:, 4]


class Scaler:
    def __init__(self, x, scale="abs", log10=False):
        self.scale = scale
        if np.isscalar(log10):
            self.log10 = np.full(x.shape[1], log10)
        else:
            self.log10 = log10
        self.useLog10 = np.any(self.log10)

        self.min = self.max = self.median = self.std = None
        self.is_trained = False

        if x is not None:
            self.train(x)

    def train(self, x):
        if self.useLog10:
            x = np.copy(x)
            x = np.log10(x, where=self.log10, out=x)
        if self.scale == "abs":
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)
        elif self.scale == "std":
            self.std = np.std(x, axis=0)
            self.median = np.median(x, axis=0)
        else:
            raise ValueError("Scale parameter is not understood")
        self.is_trained = True

    def apply(self, x):
        if not self.is_trained:
            raise ValueError("Scaler needs to be trained first")
        if self.useLog10:
            x = np.copy(x)
            x = np.log10(x, where=self.log10, out=x)
        if self.scale == "abs":
            return (x - self.min) / (self.max - self.min)
        elif self.scale == "std":
            return (x - self.median) / self.std
        else:
            raise ValueError("Scale parameter is not understood")

    def unscale(self, x):
        if not self.is_trained:
            raise ValueError("Scaler needs to be trained first")
        if self.scale == "abs":
            x = x * (self.max - self.min) + self.min
        elif self.scale == "std":
            x = x * self.std + self.median
        else:
            raise ValueError("Scale parameter is not understood")
        if self.useLog10:
            x = np.copy(x)
            x = np.power(10.0, x, where=self.log10, out=x)
        return x

    def save(self, fname):
        """ Save the scaler information to a file """
        if self.scale == "abs":
            arr0, arr1 = self.min, self.max
        elif self.scale == "std":
            arr0, arr1 = self.median, self.std
        else:
            raise ValueError

        np.savez(fname, scale=self.scale, log10=self.log10, arr0=arr0, arr1=arr1)

    @staticmethod
    def load(fname):
        data = np.load(fname)

        scale = data["scale"]
        log10 = data["log10"]
        arr0, arr1 = data["arr0"], data["arr1"]
        scaler = Scaler(None, scale=scale, log10=log10)

        if scale == "abs":
            scaler.min, scaler.max = arr0, arr1
        elif scale == "std":
            scaler.median, scaler.std = arr0, arr1
        else:
            raise ValueError

        if arr0 is not None or arr1 is not None:
            scaler.is_trained = True

        return scaler


class SME_Atmo_Model:
    def __init__(
        self,
        nlayers=4,
        normalize=False,
        log10=True,
        activation="sigmoid",
        optimizer="adam",
        parameters=["TEMP", "RHO", "RHOX", "XNA", "XNE"],
    ):
        self.nlayers = nlayers
        self.normalize = normalize
        self.log10 = log10
        self.activation = activation
        self.optimizer = optimizer
        self.labels = ["TEFF", "LOGG", "MONH"]
        self.parameters = parameters
        self.lfs = None  # TODO setup_lfs()
        self.scaler_x = None
        self.scaler_y = None

        if log10:
            self.idx_y_logcols = np.array(self.parameters) != "TEMP"
        else:
            self.idx_y_logcols = np.full(len(self.parameters), False)

        self.dim_in = len(self.labels) + 1
        self.dim_out = len(self.parameters)

        self.model = None
        self.create_model()

    @property
    def output_dir(self):
        output_dir = f"{self.nlayers}layers_"
        output_dir += "-".join(self.parameters)

        if self.log10:
            output_dir += "_log10"
        if self.normalize:
            output_dir += "_norm"
        else:
            output_dir += "_std"

        output_dir += f"_{self.activation}_{self.optimizer}"

        output_dir = join(dirname(__file__), output_dir)
        return output_dir

    @property
    def outmodel_file(self):
        return join(self.output_dir, "model_weights.h5")

    def get_tau(self, resolution=1):
        points = [11, 40, 5]
        points = [p * resolution for p in points]
        tau1 = np.linspace(-5, -3, points[0])
        tau2 = np.linspace(-2.9, 1.0, points[1])
        tau3 = np.linspace(1.2, 2, points[2])
        tau = np.concatenate((tau1, tau2, tau3))
        return tau

    def load_data(self, atmo_name):
        fname = self.lfs.get(atmo_name)
        data = readsav(fname)["atmo_grid"]

        tau = self.get_tau()
        tau_test = self.get_tau(3)

        # Define the input data (here use temperature)
        x_raw = np.array([data[ln] for ln in self.labels]).T
        y_raw = [np.concatenate(data[pa]).reshape(-1, 56) for pa in self.parameters]

        inputs = []
        outputs = []
        inputs_test = []
        outputs_test = []
        for i, x_row in enumerate(x_raw):
            for k, tau_val in enumerate(tau):
                inputs.append([x_row[0], x_row[1], x_row[2], tau_val])
                out_row = []
                for i_par in range(len(self.parameters)):
                    out_row.append(y_raw[i_par][i][k])
                outputs.append(out_row)

            for i_par in range(len(self.parameters)):
                outputs_test.append(np.interp(tau_test, tau, y_raw[i_par][i]))

        for x_row in x_raw:
            for tau_val in tau_test:
                inputs_test.append([x_row[0], x_row[1], x_row[2], tau_val])

        x_train = np.array(inputs)
        x_test = np.array(inputs_test)
        y_train = np.array(outputs)
        y_test = np.array(outputs_test).reshape(len(self.parameters), -1).T

        return (x_train, y_train, x_test, y_test)

    def extract_samples(self, x, y, n=1):
        parametersets = np.unique(x[:, :3], axis=0)
        idx = np.random.choice(parametersets.shape[0], replace=False, size=n)
        mask = np.all(x[:, :3] == parametersets[idx], axis=1)

        x_test, y_test = x[mask], y[mask]
        x_train, y_train = x[~mask], y[~mask]

        return x_train, y_train, x_test, y_test

    def create_model(self):
        # create ann model
        self.model = Sequential()
        layers = [50, 200, 350, 500, 500, 350, 200, 50]
        n = 1
        self.model.add(
            Dense(
                layers[0],
                input_shape=(self.dim_in,),
                activation=self.activation,
                name="E_in",
            )
        )

        self.model.add(Dense(layers[1], activation=self.activation, name=f"E_{n}"))
        n += 1

        if self.nlayers >= 5:
            self.model.add(Dense(layers[2], activation=self.activation, name=f"E_{n}"))
        n += 1

        if self.nlayers >= 7:
            self.model.add(Dense(layers[3], activation=self.activation, name=f"E_{n}"))
        n += 1

        if self.nlayers >= 8:
            self.model.add(Dense(layers[-4], activation=self.activation, name=f"E_{n}"))
        n += 1

        if self.nlayers >= 6:
            self.model.add(Dense(layers[-3], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(layers[-2], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(layers[-1], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(self.dim_out, activation="linear", name="E_out"))
        # self.model.summary()

        return self.model

    def train_scaler(self, x_train, y_train):
        scale = "abs" if self.normalize else "std"
        self.scaler_x = Scaler(scale=scale, x=x_train)
        self.scaler_y = Scaler(scale=scale, x=y_train, log10=self.idx_y_logcols)

    def train_model(self, x_train, y_train, x_test, y_test):
        self.train_scaler(x_train, y_train)
        x_train = self.scaler_x.apply(x_train)
        y_train = self.scaler_y.apply(y_train)

        self.model.compile(optimizer=self.optimizer, loss="mse")
        ann_fit_hist = self.model.fit(
            x_train,
            y_train,
            epochs=1000 + self.nlayers * 500,
            shuffle=True,
            batch_size=4096,
            validation_split=0.1,
            # validation_data=(x_test, y_test),
            verbose=2,
        )
        self.save_model()

        np.savez(
            join(self.output_dir, "history.npz"),
            loss=ann_fit_hist.history["loss"],
            val_loss=ann_fit_hist.history["val_loss"],
        )

        plt.figure(figsize=(13, 5))
        plt.plot(ann_fit_hist.history["loss"], label="Train", alpha=0.5)
        plt.plot(ann_fit_hist.history["val_loss"], label="Validation", alpha=0.5)
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        # plt.ylim(0.0, min(1, np.nanpercentile(ann_fit_hist.history["loss"], 96)))
        plt.xlim(-5, max(1, len(ann_fit_hist.history["loss"])) + 5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(join(self.output_dir, "sme_ann_network_loss.png"), dpi=300)
        plt.close()
        return self.model

    def load_model(self, fname):
        output_dir = dirname(fname)
        self.model.load_weights(fname, by_name=True)
        self.scaler_x = Scaler.load(join(output_dir, "scaler_x.npz"))
        self.scaler_y = Scaler.load(join(output_dir, "scaler_y.npz"))
        return self.model

    def save_model(self, fname):
        output_dir = dirname(fname)
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_weights(fname)
        self.scaler_x.save(join(output_dir, "scaler_x.npz"))
        self.scaler_y.save(join(output_dir, "scaler_y.npz"))

    def predict(self, x):
        xt = self.scaler_x.apply(x)
        yt = self.model.predict(xt, verbose=2, batch_size=2048)
        y = self.scaler_y.unscale(yt)
        return y

    def plot_results(self, x_train, y_train, x_test, y_test, y_new, y_test_new):

        n_params = len(self.parameters)
        teff_vals, logg_vals, meh_vals = np.unique(x_test[:, :3], axis=0).T
        print(teff_vals, logg_vals, meh_vals)

        for u_teff in teff_vals:
            for u_logg in logg_vals:
                for u_meh in meh_vals:

                    # idx_plot = (
                    #     (x_train[:, 0] == u_teff)
                    #     & (x_train[:, 1] == u_logg)
                    #     & (x_train[:, 2] == u_meh)
                    # )

                    idx_plot = (
                        (x_test[:, 0] == u_teff)
                        & (x_test[:, 1] == u_logg)
                        & (x_test[:, 2] == u_meh)
                    )

                    # if np.sum(idx_plot) <= 0:
                    #     continue

                    print("Plot points:", np.sum(idx_plot), u_teff, u_logg, u_meh)

                    fig, ax = plt.subplots(
                        n_params, 1, sharex=True, figsize=(7, 2.5 * n_params)
                    )

                    if n_params == 1:
                        ax = [ax]

                    for i_par in range(n_params):
                        ax[i_par].plot(
                            x_test[idx_plot, 3],
                            y_test[:, i_par][idx_plot],
                            label="Reference",
                            alpha=0.5,
                        )
                        ax[i_par].plot(
                            x_test[idx_plot, 3],
                            y_test_new[:, i_par][idx_plot],
                            label="Predicted",
                            alpha=0.5,
                            ls="--",
                        )
                        ax[i_par].set(ylabel=self.parameters[i_par])

                    ax[-1].set(xlabel="tau")
                    ax[0].legend()
                    fig.tight_layout()
                    fig.subplots_adjust(wspace=0.0)
                    fname = f"{u_teff:.1f}-teff_{u_logg:.1f}-logg_{u_meh:.1f}-monh.png"
                    fname = join(self.output_dir, fname)
                    fig.savefig(fname)
                    plt.close()


def interpolate_grid(atmo, teff, logg, monh, lfs_atmo=None, verbose=0):
    nlayers = 4
    normalize = False
    log10_params = True
    activation = "sigmoid"
    optimizer = "adam"
    parameters = ["TEMP", "RHO", "RHOX", "XNA", "XNE"]

    # TODO: load correct model
    fname = atmo.source

    # TODO: persist model in memory between calls
    model = SME_Atmo_Model(
        nlayers, normalize, log10_params, activation, optimizer, parameters=parameters
    )

    model.load_model(model.outmodel_file)

    tau = model.get_tau()
    # x = ["TEFF", "LOGG", "MONH", "TAU"]
    x = np.zeros((len(tau), 4))
    x[:, :3] = teff, logg, monh
    x[:, 3] = tau

    # y = ["TEMP", "RHO", "RHOX", "XNA", "XNE"]
    y = model.predict(x)

    atmo.tau = tau
    atmo.temp = y[:, 0]
    atmo.rho = y[:, 1]
    atmo.rhox = y[:, 2]
    atmo.xna = y[:, 3]
    atmo.xne = y[:, 4]

    return atmo
