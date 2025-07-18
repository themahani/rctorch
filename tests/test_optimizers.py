import numpy as np
import pytest
import torch
import torch.nn as nn

from rctorch import Reservoir
from rctorch.models import LIF
from rctorch.optimizers import BruteForceMesh
from rctorch.supervisors import LorenzAttractor
from rctorch.utils import z_transform


@pytest.fixture
def model_params():
    Ne = 50
    Ni = 50
    N = Ne + Ni
    dt = 0.1
    current = np.ones((N, 1)) * 75
    return {
        "dt": dt,
        "BIAS": current,
        "Ne": Ne,
        "Ni": Ni,
        "device": torch.device("cpu"),
    }


@pytest.fixture
def lorenz_supervisor(model_params):
    dt = model_params["dt"]
    T = 500
    tau = 0.02
    x = LorenzAttractor(T, dt, tau).generate(transient_time=50)
    x_tensor = torch.tensor(z_transform(x.T), dtype=torch.float32, device=model_params["device"])
    return x_tensor


@pytest.fixture
def reservoir_params(lorenz_supervisor):
    n_input = lorenz_supervisor.size(0)
    n_output = n_input
    return {
        "model_cls": LIF,
        "n_input": n_input,
        "n_output": n_output,
    }


@pytest.fixture
def bfm_params():
    w_in_amp_range = np.linspace(5, 20, 2)
    gbar_range = np.linspace(5, 10, 2)
    return {
        "w_in_amp": w_in_amp_range,
        "gbar": gbar_range,
    }


@pytest.fixture
def render_kwargs(lorenz_supervisor):
    nt_transient = 20
    ridge_ = 1.0
    ff_coef = 1.0
    return {
        "x": lorenz_supervisor,
        "nt_transient": nt_transient,
        "ridge_reg": ridge_,
        "ff_coef": ff_coef,
    }


def test_bruteforcemesh(reservoir_params, model_params, render_kwargs, bfm_params):
    model_params.pop("dt")
    total_params = reservoir_params | model_params
    bfm = BruteForceMesh(total_params, render_kwargs, bfm_params, num_threads=2)
    bfm.run("./test_optimizers")
