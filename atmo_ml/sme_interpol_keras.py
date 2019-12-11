"""
Interpolate on the SME atmosphere grid
using machine learning methods

Credit to Klemen Cotar
"""
import os, sys
from os.path import expanduser, join, dirname

os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib

matplotlib.use("Agg")
import tensorflow as tf

tf.config.optimizer.set_jit(True)

import numpy as np
from scipy.io import readsav
import joblib
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.layers.advanced_activations import PReLU, ReLU
from astropy.table import Table

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


lfs = setup_lfs()


def get_tau():
    tau1 = np.linspace(-5, -3, 11)
    tau2 = np.linspace(-2.9, 1.0, 40)
    tau3 = np.linspace(1.2, 2, 5)
    tau = np.concatenate((tau1, tau2, tau3))
    return tau


def get_tau_test():
    tau1 = np.linspace(-5, -3, 30)
    tau2 = np.linspace(-2.9, 1.0, 120)
    tau3 = np.linspace(1.2, 2, 30)
    tau = np.concatenate((tau1, tau2, tau3))
    return tau


"""
Run scripts

nohup python sme_interpol_keras.py > /dev/null &
nohup python sme_interpol_keras.py TEMP > /dev/null &
nohup python sme_interpol_keras.py RHO > /dev/null &
nohup python sme_interpol_keras.py RHOX > /dev/null &
nohup python sme_interpol_keras.py XNA > /dev/null &
nohup python sme_interpol_keras.py XNE > /dev/null &

nohup python sme_interpol_keras.py > all.log &
nohup python sme_interpol_keras.py TEMP > TEMP.log &
nohup python sme_interpol_keras.py RHO > RHO.log &
nohup python sme_interpol_keras.py RHOX > RHOX.log &
nohup python sme_interpol_keras.py XNA > XNA.log &
nohup python sme_interpol_keras.py XNE > XNE.log &

"""


def load_data(labels, parameters, fname, NORMALIZE, log10_params):
    fname = lfs.get(fname)
    data = readsav(fname)["atmo_grid"]

    tau = get_tau()
    tau_test = get_tau_test()

    # Define the input data (here use temperature)
    x_raw = np.array([data[ln] for ln in labels]).T
    y_raw = [data[pa] for pa in parameters]

    labels = []
    labels_test = []
    output = []
    for i, x_row in enumerate(x_raw):
        for k, tau_val in enumerate(tau):
            labels.append([x_row[0], x_row[1], x_row[2], tau_val])
            out_row = []
            for i_par in range(len(parameters)):
                out_row.append(y_raw[i_par][i][k])
            output.append(out_row)

    for x_row in x_raw:
        for tau_val in tau_test:
            labels_test.append([x_row[0], x_row[1], x_row[2], tau_val])

    x = np.array(labels)
    x_test = np.array(labels_test)
    y = np.array(output)

    # logaritmize output parameters
    if log10_params:
        # logaritmize all output parameters except TEMP that is already has manageable span og values
        idx_y_logcols = np.array(parameters) != "TEMP"
        y[:, idx_y_logcols] = np.log10(y[:, idx_y_logcols])

    # scale output values
    y_max = np.max(y, axis=0)
    y_min = np.min(y, axis=0)
    y_median = np.median(y, axis=0)
    y_std = np.std(y, axis=0)

    # normalization or standardization of outputs
    if NORMALIZE:
        y_train = (y - y_min) / (y_max - y_min)
    else:
        y_train = (y - y_median) / y_std

    print(x)

    # scale input labels
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_median = np.median(x, axis=0)
    x_std = np.std(x, axis=0)
    # normalization or standardization of inputs
    if NORMALIZE:
        x_train = (x - x_min) / (x_max - x_min)
        x_test_train = (x_test - x_min) / (x_max - x_min)
    else:
        x_train = (x - x_median) / x_std
        x_test_train = (x_test - x_median) / x_std

    print("X ranges:", x_min, x_max)

    return (
        labels,
        labels_test,
        output,
        x_train,
        y_train,
        x_test_train,
        y_train,
        y_min,
        y_max,
        y_median,
        y_std,
        idx_y_logcols,
    )


def prepare_output(
    parameters, n_layers, activation, optimizer, log10_params, NORMALIZE
):
    output_dir = str(n_layers) + "layers"
    output_dir += "_" + "-".join(parameters)

    if log10_params:
        output_dir += "_log10"
    if NORMALIZE:
        output_dir += "_norm"
    else:
        output_dir += "_std"

    output_dir += "_" + str(activation)
    output_dir += "_" + str(optimizer)

    output_dir = join(dirname(__file__), output_dir)

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    return output_dir


def create_model(dim_in, dim_out, n_layers, activation=None):
    # create ann model
    sme_nn = Sequential()

    sme_nn.add(Dense(50, input_shape=(dim_in,), activation=activation, name="E_in"))
    if activation is None:
        sme_nn.add(PReLU(name="PR_0"))

    sme_nn.add(Dense(200, activation=activation, name="E_1"))
    if activation is None:
        sme_nn.add(PReLU(name="PR_1"))

    if n_layers >= 5:
        sme_nn.add(Dense(350, activation=activation, name="E_2"))
        if activation is None:
            sme_nn.add(PReLU(name="PR_2"))

    if n_layers >= 7:
        sme_nn.add(Dense(500, activation=activation, name="E_3"))
        if activation is None:
            sme_nn.add(PReLU(name="PR_3"))

    if n_layers >= 8:
        sme_nn.add(Dense(500, activation=activation, name="E_4"))
        if activation is None:
            sme_nn.add(PReLU(name="PR_4"))

    if n_layers >= 6:
        sme_nn.add(Dense(350, activation=activation, name="E_5"))
        if activation is None:
            sme_nn.add(PReLU(name="PR_5"))

    sme_nn.add(Dense(200, activation=activation, name="E_6"))
    if activation is None:
        sme_nn.add(PReLU(name="PR_6"))

    sme_nn.add(Dense(50, activation=activation, name="E_7"))
    if activation is None:
        sme_nn.add(PReLU(name="PR_7"))

    sme_nn.add(Dense(dim_out, activation="linear", name="E_out"))
    sme_nn.summary()
    return sme_nn


def train_model(sme_nn, x_train, y_train, optimizer, out_model_file):
    sme_nn.compile(optimizer=optimizer, loss="mse")
    ann_fit_hist = sme_nn.fit(
        x_train,
        y_train,
        epochs=1000 + n_layers * 500,
        shuffle=True,
        batch_size=4096,
        validation_split=0,
        validation_data=(x_train, y_train),
        verbose=2,
    )
    sme_nn.save_weights(out_model_file)

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
    plt.savefig("sme_ann_network_loss.png", dpi=300)
    plt.close()
    return sme_nn


def predict_values(
    sme_nn,
    x_train,
    x_test,
    y_test,
    y_min,
    y_max,
    y_median,
    y_std,
    NORMALIZE,
    log10_params,
    idx_y_logcols,
):
    print("Predicting model values")
    y_predicted = sme_nn.predict(x_train, verbose=2, batch_size=2048)

    print("Predicting model values on denser tau grid ")
    y_test_predicted = sme_nn.predict(x_test, verbose=2, batch_size=2048)

    if NORMALIZE:
        y_new = y_predicted * (y_max - y_min) + y_min
        y_test_new = y_test_predicted * (y_max - y_min) + y_min
    else:
        y_new = y_predicted * y_std + y_median
        y_test_new = y_test_predicted * y_std + y_median

    # anti-logaritmize output parameters
    if log10_params:
        y_test[:, idx_y_logcols] = 10.0 ** y_test[:, idx_y_logcols]
        y_new[:, idx_y_logcols] = 10.0 ** y_new[:, idx_y_logcols]
        y_test_new[:, idx_y_logcols] = 10.0 ** y_test_new[:, idx_y_logcols]
    return y_test, y_new, y_test_new


def plot_results(
    labels,
    labels_test,
    output,
    x_train,
    y_train,
    x_test,
    y_test,
    y_new,
    y_test_new,
    parameters,
    output_dir,
):
    x = np.array(labels)
    x_test = np.array(labels_test)
    y = np.array(output)

    n_params = len(parameters)
    teff_vals = np.unique(x[:, 0])[::2]
    logg_vals = [3]  # np.unique(x_train[:, 1])[::2]
    meh_vals = [0]  # np.unique(x_train[:, 2])[::3]
    print(teff_vals, logg_vals, meh_vals)

    for u_teff in teff_vals:
        for u_logg in logg_vals:
            for u_meh in meh_vals:

                idx_plot = (
                    (x[:, 0] == u_teff) & (x[:, 1] == u_logg) & (x[:, 2] == u_meh)
                )

                idx_plot_test = (
                    (x_test[:, 0] == u_teff)
                    & (x_test[:, 1] == u_logg)
                    & (x_test[:, 2] == u_meh)
                )

                if np.sum(idx_plot) <= 0:
                    continue

                print("Plot points:", np.sum(idx_plot), u_teff, u_logg, u_meh)
                plot_x = x[:, 3][idx_plot]
                plot_x_test = x_test[:, 3][idx_plot_test]

                fig, ax = plt.subplots(
                    n_params, 1, sharex=True, figsize=(7, 2.5 * n_params)
                )

                if n_params == 1:
                    ax = [ax]

                for i_par in range(n_params):
                    ax[i_par].plot(
                        plot_x, y[:, i_par][idx_plot], label="Reference", alpha=0.5
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
                    ax[i_par].set(ylabel=parameters[i_par])

                ax[-1].set(xlabel="tau")
                ax[0].legend()
                fig.tight_layout()
                fig.subplots_adjust(wspace=0.0)
                fname = f"{u_teff:.1f}-teff_{u_logg:.1f}-logg_{u_meh:.1f}-monh.png"
                fname = join(output_dir, fname)
                fig.savefig(fname)
                plt.close()


def run_nn(in_args, NORMALIZE, activation, optimizer, n_layers=4, log10_params=True):
    """
    Train the neural network
    """

    # -----------------------------------------------------------------------------
    fname = "marcs2012p_t0.0.sav"
    labels = ["TEFF", "LOGG", "MONH"]
    parameters = ["TEMP", "RHO", "RHOX", "XNA", "XNE"]

    if len(in_args) >= 2:
        parameters = in_args[1].split("_")
    print("Parameters:", parameters)

    # Load data
    labels, labels_test, output, x_train, y_train, x_test, y_test, y_min, y_max, y_median, y_std, idx_y_logcols = load_data(
        labels, parameters, fname, NORMALIZE, log10_params
    )

    # Prepare output folder
    output_dir = prepare_output(
        parameters, n_layers, activation, optimizer, log10_params, NORMALIZE
    )

    # KERAS NN implementation
    print("Dimesions labels:", x_train.shape, y_train.shape)
    dim_in = x_train.shape[1]
    dim_out = y_train.shape[1]
    sme_nn = create_model(dim_in, dim_out, n_layers, activation)

    # Load or train weights
    out_model_file = join(output_dir, "model_weights.h5")
    if os.path.isfile(out_model_file):
        sme_nn.load_weights(out_model_file, by_name=True)
    else:
        sme_nn = train_model(sme_nn, x_train, y_train, optimizer, out_model_file)

    # Predict output
    y_test, y_new, y_test_new = predict_values(
        sme_nn,
        x_train,
        x_test,
        y_test,
        y_min,
        y_max,
        y_median,
        y_std,
        NORMALIZE,
        log10_params,
        idx_y_logcols,
    )

    # Plot comparison
    plot_results(
        labels,
        labels_test,
        output,
        x_train,
        y_train,
        x_test,
        y_test,
        y_new,
        y_test_new,
        parameters,
        output_dir,
    )


if __name__ == "__main__":
    n_layers = 4
    log_y_values = True
    u_NORMALIZE = False
    u_activation = "sigmoid"
    u_optimizer = "adam"

    run_nn(
        sys.argv,
        u_NORMALIZE,
        u_activation,
        u_optimizer,
        n_layers=n_layers,
        log10_params=log_y_values,
    )
