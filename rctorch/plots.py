#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models.morris_lecar import MorrisLecar
import os


def plot_model(
    model: MorrisLecar,
    prefix: str,
    neurons: torch.Tensor,
    v_trace: torch.Tensor,
    decoders: torch.Tensor,
    rls_start: float,
    rls_stop: float,
    rls_step: int,
    lamda: float,
    Q: float,
    n_vars: int = 3,
    mode: str = "vertical",
    save_dir: str = "img",
):
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    x_hat_rec = model.x_hat_rec.cpu()
    # Convert all data to numpy arrays
    v_trace = v_trace.numpy()
    sup = model.sup.cpu().numpy()

    t = model.time.cpu().numpy()
    x = model.sup.cpu().numpy()
    N = model._N
    Ie = model._BIAS[0, 0]
    Ii = model._BIAS[-1, 0]

    # Plot the Voltage Trace
    fig, ax = plt.subplots(figsize=(20, 6))
    for i in range(len(neurons)):
        signal = v_trace[:, i]
        minim = np.nanmin(signal)
        maxim = np.nanmax(signal)
        signal = (signal - minim) / (maxim - minim) + i
        ax.plot(t, signal)
    plt.grid(alpha=0.5)
    plt.title(f"N={N}, Ie={Ie}, Ii={Ii}, Q={Q}, l={lamda}, RLS step={rls_step}")
    plt.savefig(os.path.join(save_dir, prefix + "_voltage_trace.jpg"), bbox_inches="tight", dpi=300)
    plt.close()

    # Plot the RLS results
    nrows = np.min(list(x.shape))
    fig, ax = plt.subplots(figsize=(20, 15), nrows=nrows, sharex=True)
    plt.suptitle(f"N={N}, Ie={Ie}, Ii={Ii}, Q={Q}, l={lamda}, RLS step={rls_step}")
    for i in range(nrows):
        signal = sup[:, i]
        ax[i].plot(t, signal, "b", label="supervisor")
        ax[i].plot(t, x_hat_rec[:, i], "g", label="decoded")
        ax[i].set_ylim(np.min(signal) - 1, np.max(signal) + 1)
        ax[i].grid(alpha=0.5)
        ax[i].legend(loc=0)
        ax[i].axvline(x=rls_start, c="r", label="start RLS")
        ax[i].axvline(x=rls_stop, c="magenta", label="stop RLS")
    plt.savefig(os.path.join(save_dir, prefix + "_output_test.jpg"), bbox_inches="tight", dpi=300)
    plt.close()

    if n_vars > 1:
        labels = ["x", "y", "z", "w", "v", "p", "q"]
        nrows = n_vars
        labels = labels[:nrows]

        dt = t[1] - t[0]
        rls_stp = round(rls_stop // dt)

        if mode == "vertical":
            fig, ax = plt.subplots(figsize=(5, 15), nrows=nrows)
        elif mode == "horizontal":
            fig, ax = plt.subplots(figsize=(15, 5), ncols=nrows)
        else:
            raise ValueError("mode should either be 'vertical' or 'horizontal'!")

        plt.suptitle("Phase Space")
        for i in range(nrows):
            ax[i].plot(
                x_hat_rec[rls_stp:, i - 1],
                x_hat_rec[rls_stp:, i],
                "b",
                lw=1,
                label="estimator",
            )
            ax[i].plot(sup[rls_stp:, i - 1], sup[rls_stp:, i], "g", lw=1, label="supervisor")
            ax[i].legend(loc=0)
            ax[i].set_xlabel(labels[i - 1])
            ax[i].set_ylabel(labels[i])
        plt.savefig(os.path.join(save_dir, prefix + "_portrait_xyz.jpg"), bbox_inches="tight", dpi=300)
        plt.close()

    fig, ax = plt.subplots(figsize=(20, 6))
    for i in range(len(neurons)):
        plt.plot(t, decoders[:, i], label=f"decoder {i}")
    plt.grid(alpha=0.5)
    plt.axvline(x=rls_start, c="r", label="start RLS")
    plt.axvline(x=rls_stop, c="magenta", label="stop RLS")
    plt.legend(loc=0)
    plt.suptitle(f"N={N}, Ie={Ie}, Ii={Ii}, Q={Q}, l={lamda}, RLS step={rls_step}")
    plt.savefig(os.path.join(save_dir, prefix + "_test_decoders.jpg"), bbox_inches="tight", dpi=300)
    plt.close()
