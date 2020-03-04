""" Handles reading and interpolation of atmopshere (grid) data """
import itertools
import logging
import os
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from scipy.interpolate import interp1d
from scipy.io import readsav
from scipy.optimize import curve_fit
from tqdm import tqdm

from ..large_file_storage import setup_atmo
from .atmosphere import Atmosphere as Atmo
from .atmosphere import AtmosphereError
from .savfile import SavFile

tf.config.optimizer.set_jit(True)

logger = logging.getLogger(__name__)


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
        self.lfs = setup_atmo()
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

    def load_data(self, atmo, nextra=5000, ntest=1000):
        fname = self.lfs.get(atmo.source)
        data = readsav(fname)["atmo_grid"]

        tau = self.get_tau()
        ntau = len(tau)

        # Define the input data (here use temperature)
        x_raw = np.array([data[ln] for ln in self.labels]).T
        y_raw = [np.concatenate(data[pa]).reshape(-1, ntau) for pa in self.parameters]

        inputs = []
        outputs = []
        for i, x_row in enumerate(x_raw):
            for k, tau_val in enumerate(tau):
                inputs.append([x_row[0], x_row[1], x_row[2], tau_val])
                out_row = []
                for i_par in range(len(self.parameters)):
                    out_row.append(y_raw[i_par][i][k])
                outputs.append(out_row)

        x_train = np.array(inputs)
        y_train = np.array(outputs)

        # Create additional training, test, and validation data from the traditional interpolation
        npoints = nextra + ntest
        teff_min, logg_min, monh_min, _ = x_train.min(axis=0)
        teff_max, logg_max, monh_max, _ = x_train.max(axis=0)

        x_test = np.zeros((npoints * (ntau - 1), 4))
        y_test = np.zeros((npoints * (ntau - 1), y_train.shape[1]))

        logger.disabled = True

        for i in tqdm(range(npoints), desc="Dataset"):
            # We have to make sure the interpolation succeeded
            passed = False
            while not passed:
                t = teff_min + np.random.rand() * (teff_max - teff_min)
                l = logg_min + np.random.rand() * (logg_max - logg_min)
                m = monh_min + np.random.rand() * (monh_max - monh_min)
                try:
                    atmo_interp = interp_atmo_grid(t, l, m, atmo, self.lfs)
                    passed = True
                except AtmosphereError:
                    pass

            x_test[i * (ntau - 1) : (i + 1) * (ntau - 1), 3] = np.log10(
                atmo_interp["TAU"]
            )
            x_test[i * (ntau - 1) : (i + 1) * (ntau - 1), :3] = t, l, m

            for j, par in enumerate(self.parameters):
                y_test[i * (ntau - 1) : (i + 1) * (ntau - 1), j] = atmo_interp[par]

        logger.disabled = False

        x_train = np.concatenate((x_train, x_test[: nextra * (ntau - 1)]))
        y_train = np.concatenate((y_train, y_test[: nextra * (ntau - 1)]))
        x_test = x_test[nextra * (ntau - 1) :]
        y_test = y_test[nextra * (ntau - 1) :]

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
        self.save_model(self.outmodel_file)

        np.savez(
            join(self.output_dir, "history.npz"),
            loss=ann_fit_hist.history["loss"],
            val_loss=ann_fit_hist.history["val_loss"],
        )
        self.plot_history(ann_fit_hist)
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

    def plot_history(self, history):
        plt.figure(figsize=(13, 5))
        plt.plot(history.history["loss"], label="Train", alpha=0.5)
        plt.plot(history.history["val_loss"], label="Validation", alpha=0.5)
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.xlim(-5, max(1, len(history.history["loss"])) + 5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(join(self.output_dir, "sme_ann_network_loss.png"), dpi=300)
        plt.close()

    def plot_results(self, x_train, y_train, x_test, y_test, y_new, y_test_new):
        n_params = len(self.parameters)
        x_plot = np.unique(x_test[:, :3], axis=0)

        for x in x_plot:
            u_teff, u_logg, u_meh = x
            idx_plot = np.all(x_test[:, :3] == x, axis=1)
            idx_plot_train = x_train[:, 0] == u_teff

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


def train_grid(atmo):
    nlayers = 4
    normalize = False
    log10_params = True
    activation = "sigmoid"
    optimizer = "adam"
    parameters = ["TEMP", "RHO", "RHOX", "XNA", "XNE"]

    model = SME_Atmo_Model(
        nlayers, normalize, log10_params, activation, optimizer, parameters=parameters
    )

    # Replace the training and sample creation with this if data exists
    # data = np.load("test_data.npz")
    # x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    # model.load_model(model.outmodel_file)

    x_train, y_train, x_test, y_test = model.load_data(atmo, 5000, 1000)
    np.savez(
        "test_data.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

    model.train_model(x_train, y_train, x_test, y_test)

    y_new = model.predict(x_train)
    y_test_new = model.predict(x_test)

    # This will make a lot of plots
    model.plot_results(x_train, y_train, x_test, y_test, y_new, y_test_new)

    pass


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
