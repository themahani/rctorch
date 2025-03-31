import numpy as np
import pytest
import torch

from ml_force import MorrisLecar, MorrisLecarCurrent, z_transform


@pytest.fixture
def supervisor():
    T = 1000
    dt = 0.1
    t = np.arange(0, T, dt)
    return np.sin(2 * np.pi * t / 100).reshape(-1, 1)


@pytest.fixture
def model_params(supervisor):
    N = 100
    dt = 0.1
    T = 1000
    current = np.ones((N, 1)) * 75

    return {
        "supervisor": supervisor,
        "dt": dt,
        "T": T,
        "BIAS": current,
        "N": N,
        "device": torch.device("cpu"),
    }


def test_morris_lecar_initialization(model_params):
    model = MorrisLecar(**model_params)
    assert model.v.shape == (model_params["N"], 1)
    assert model.s.shape == (model_params["N"], 1)
    assert model.n.shape == (model_params["N"], 1)
    assert model.dec.shape == (model_params["N"], 1)


def test_morris_lecar_current_initialization(model_params):
    model = MorrisLecarCurrent(**model_params)
    assert model.v.shape == (model_params["N"], 1)
    assert model.s.shape == (model_params["N"], 1)
    assert model.n.shape == (model_params["N"], 1)
    assert model.dec.shape == (model_params["N"], 1)


def test_z_transform():
    x = np.array([1, 2, 3, 4, 5])
    z = z_transform(x)
    assert np.allclose(z.mean(), 0)
    assert np.allclose(z.std(), 1)


def test_morris_lecar_euler_step(model_params):
    model = MorrisLecar(**model_params)
    v_before = model.v.clone()
    s_before = model.s.clone()
    n_before = model.n.clone()

    model.euler_step()

    assert not torch.allclose(v_before, model.v)
    assert not torch.allclose(s_before, model.s)
    assert not torch.allclose(n_before, model.n)


def test_morris_lecar_render(model_params):
    model = MorrisLecar(**model_params)
    render_args = {
        "rls_start": 100,
        "rls_stop": 800,
        "rls_step": 20,
        "live_plot": False,
        "plt_interval": 100,
        "n_neurons": 10,
        "save_all": False,
    }

    neurons, v_trace, dec_trace = model.render(**render_args)

    assert neurons.shape[0] == render_args["n_neurons"]
    assert v_trace.shape[1] == render_args["n_neurons"]
    assert dec_trace.shape[1] == render_args["n_neurons"]
