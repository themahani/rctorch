import numpy as np
import pytest
import torch
import torch.nn as nn

from rctorch import Reservoir
from rctorch.models import LIF
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
def reservoir_echo_state_params():
    n_input = 1
    n_ouput = 2
    w_in_amp = 100
    return {
        "model_cls": LIF,
        "n_input": n_input,
        "n_output": n_ouput,
        "w_in_amp": w_in_amp,
    }


@pytest.fixture
def lorenz_supervisor(model_params):
    dt = model_params["dt"]
    T = 500
    tau = 0.02
    x = LorenzAttractor(T, dt, tau).generate(transient_time=50)
    x_tensor = torch.tensor(z_transform(x.T), dtype=torch.float32, device=model_params["device"])
    return x_tensor


def test_resrvoir(reservoir_echo_state_params, model_params, lorenz_supervisor):
    res = Reservoir(**reservoir_echo_state_params, **model_params)
    assert res.W_in.shape == (res.n_hidden, res.n_input)
    assert res.W_out.shape == (res.n_hidden, res.n_output)

    x_axes = [0]
    y_axes = [1, 2]
    x, y = lorenz_supervisor[:, x_axes], lorenz_supervisor[:, y_axes]
    assert x.size(1) == res.n_input
    assert y.size(1) == res.n_output

    y_hat = res.fit_echo_state(x, y, nt_transient=20)
    assert y_hat.size() == y.size()  # Dimensions of y_hat and y should match
