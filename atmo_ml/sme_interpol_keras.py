"""
Interpolate on the SME atmosphere grid
using machine learning methods

Credit to Klemen Cotar
"""
import os
import sys
from os.path import dirname, expanduser, join

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.io import readsav

# matplotlib.use("Agg")

import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import PReLU, ReLU
from keras.models import Model, Sequential, load_model

tf.config.optimizer.set_jit(True)

from pysme.config import Config
from pysme.large_file_storage import LargeFileStorage


def setup_lfs():
    config = Config()
    server = config["data.file_server"]
    storage = config["data.atmospheres"]
    cache = config["data.cache.atmospheres"]
    pointers = config["data.pointers.atmospheres"]
    lfs_atmo = LargeFileStorage(server, pointers, storage, cache)
    return lfs_atmo


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


class SME_Model:
    def __init__(
        self,
        fname,
        nlayers=4,
        normalize=False,
        log10=True,
        activation="sigmoid",
        optimizer="adam",
    ):
        self.fname = fname
        self.nlayers = nlayers
        self.normalize = normalize
        self.log10 = log10
        self.activation = activation
        self.optimizer = optimizer
        self.labels = ["TEFF", "LOGG", "MONH"]
        self.parameters = ["TEMP", "RHO", "RHOX", "XNA", "XNE"]
        self.lfs = setup_lfs()
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

    def load_data(self):
        fname = self.lfs.get(self.fname)
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

        if n_layers >= 5:
            self.model.add(Dense(layers[2], activation=self.activation, name=f"E_{n}"))
        n += 1

        if n_layers >= 7:
            self.model.add(Dense(layers[3], activation=self.activation, name=f"E_{n}"))
        n += 1

        if n_layers >= 8:
            self.model.add(Dense(layers[-4], activation=self.activation, name=f"E_{n}"))
        n += 1

        if n_layers >= 6:
            self.model.add(Dense(layers[-3], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(layers[-2], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(layers[-1], activation=self.activation, name=f"E_{n}"))
        n += 1

        self.model.add(Dense(self.dim_out, activation="linear", name="E_out"))
        self.model.summary()

        return self.model

    def train_scaler(self, x_train, y_train):
        scale = "abs" if self.normalize else "std"
        self.scaler_x = Scaler(scale=scale, x=x_train)
        self.scaler_y = Scaler(scale=scale, x=y_train, log10=self.idx_y_logcols)

    def train_model(self, x_train, y_train):
        self.train_scaler(x_train, y_train)
        x_train = self.scaler_x.apply(x_train)
        y_train = self.scaler_y.apply(y_train)

        self.model.compile(optimizer=self.optimizer, loss="mse")
        ann_fit_hist = self.model.fit(
            x_train,
            y_train,
            epochs=1000 + n_layers * 500,
            shuffle=True,
            batch_size=4096,
            validation_split=0,
            validation_data=(x_train, y_train),
            verbose=2,
        )
        self.save_model()

        plt.figure(figsize=(13, 5))
        plt.plot(ann_fit_hist.history["loss"], label="Train", alpha=0.5)
        plt.plot(ann_fit_hist.history["val_loss"], label="Validation", alpha=0.5)
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.ylim(0.0, min(1, np.nanpercentile(ann_fit_hist.history["loss"], 96)))
        plt.xlim(-5, max(1, len(ann_fit_hist.history["loss"])) + 5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(join(self.output_dir, "sme_ann_network_loss.png"), dpi=300)
        plt.close()
        return self.model

    def load_model(self):
        self.model.load_weights(self.outmodel_file, by_name=True)
        return self.model

    def save_model(self):
        self.model.save_weights(self.outmodel_file)

    def predict(self, x):
        xt = self.scaler_x.apply(x)
        yt = self.model.predict(xt, verbose=2, batch_size=2048)
        y = self.scaler_y.unscale(yt)
        return y

    def plot_results(self, x_train, y_train, x_test, y_test, y_new, y_test_new):

        n_params = len(self.parameters)
        teff_vals = np.unique(x_train[:, 0])[::2]
        logg_vals = [3]  # np.unique(x_train[:, 1])[::2]
        meh_vals = [0]  # np.unique(x_train[:, 2])[::3]
        print(teff_vals, logg_vals, meh_vals)

        for u_teff in teff_vals:
            for u_logg in logg_vals:
                for u_meh in meh_vals:

                    idx_plot = (
                        (x_train[:, 0] == u_teff)
                        & (x_train[:, 1] == u_logg)
                        & (x_train[:, 2] == u_meh)
                    )

                    idx_plot_test = (
                        (x_test[:, 0] == u_teff)
                        & (x_test[:, 1] == u_logg)
                        & (x_test[:, 2] == u_meh)
                    )

                    if np.sum(idx_plot) <= 0:
                        continue

                    print("Plot points:", np.sum(idx_plot), u_teff, u_logg, u_meh)
                    plot_x = x_train[:, 3][idx_plot]
                    plot_x_test = x_test[:, 3][idx_plot_test]

                    fig, ax = plt.subplots(
                        n_params, 1, sharex=True, figsize=(7, 2.5 * n_params)
                    )

                    if n_params == 1:
                        ax = [ax]

                    for i_par in range(n_params):
                        ax[i_par].plot(
                            plot_x,
                            y_train[:, i_par][idx_plot],
                            label="Reference",
                            alpha=0.5,
                        )
                        ax[i_par].plot(
                            plot_x,
                            y_new[:, i_par][idx_plot],
                            label="Predicted",
                            alpha=0.5,
                            ls="--",
                        )
                        ax[i_par].plot(
                            plot_x_test,
                            y_test_new[:, i_par][idx_plot_test],
                            label="Predicted denser",
                            alpha=0.5,
                            ls=":",
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


def run_nn(in_args, normalize, activation, optimizer, n_layers=4, log10=True):
    fname = "marcs2012p_t0.0.sav"

    model = SME_Model(fname, n_layers, normalize, log10, activation, optimizer)
    x_train, y_train, x_test, y_test = model.load_data()
    if os.path.isfile(model.outmodel_file):
        model.train_scaler(x_train, y_train)
        model.load_model()
    else:
        model.train_model(x_train, y_train)

    y_new = model.predict(x_train)
    y_test_new = model.predict(x_test)

    model.plot_results(x_train, y_train, x_test, y_test, y_new, y_test_new)


if __name__ == "__main__":
    n_layers = 4
    log10 = True
    normalize = False
    activation = "sigmoid"
    optimizer = "adam"

    # Source of the training data
    fname = "marcs2012p_t0.0.sav"

    run_nn([], normalize, activation, optimizer, n_layers, log10)
