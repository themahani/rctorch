import numpy as np
import pytest
import torch

from rctorch.models import LIF, MorrisLecar, MorrisLecarCurrent
from rctorch.utils import z_transform


@pytest.fixture
def model_params():
    Ne = 100
    Ni = 100
    N = Ne + Ni
    dt = 0.05
    current = np.ones((N, 1)) * 75

    return {
        "dt": dt,
        "BIAS": current,
        "Ne": Ne,
        "Ni": Ni,
        "device": torch.device("cpu"),
    }


def test_morris_lecar_initialization(model_params):
    model = MorrisLecar(**model_params)
    N = model_params["Ne"] + model_params["Ni"]
    assert model.mem.shape == (N, 1)
    assert model.s.shape == (N, 1)
    assert model.n.shape == (N, 1)
    assert model.w.shape == (N, N)
    assert model.factory_kwargs["dtype"] == torch.float32  # Default dtype
    assert torch.any(torch.logical_or(model.w != 1, model.w != 0))  # Weights must be binary by default


def test_morris_lecar_current_initialization(model_params):
    model = MorrisLecarCurrent(**model_params)
    N = model_params["Ne"] + model_params["Ni"]
    assert model.mem.shape == (N, 1)
    assert model.s.shape == (N, 1)
    assert model.n.shape == (N, 1)
    assert model.w.shape == (N, N)
    assert model.factory_kwargs["dtype"] == torch.float32  # Default dtype
    assert torch.allclose(model.w.mean(), torch.Tensor(0))


def test_lif_initialization(model_params):
    model = LIF(**model_params)
    N = model_params["Ne"] + model_params["Ni"]
    assert model.mem.shape == (N, 1)
    assert model.r.shape == (N, 1)
    assert model.w.shape == (N, N)
    assert model.factory_kwargs["dtype"] == torch.float32  # Default dtype
    assert torch.allclose(model.w.mean(), torch.Tensor(0))


def test_z_transform():
    x = np.array([1, 2, 3, 4, 5])
    z = z_transform(x)
    assert np.allclose(z.mean(), 0)
    assert np.allclose(z.std(), 1)


def test_morris_lecar_forward(model_params):
    model = MorrisLecar(**model_params)
    v_before = model.mem.clone()
    s_before = model.s.clone()
    n_before = model.n.clone()
    input = torch.zeros(size=model.mem.size(), dtype=torch.float32, device=model_params["device"])
    model.forward(input)

    assert not torch.allclose(v_before, model.mem)
    assert not torch.allclose(s_before, model.s)
    assert not torch.allclose(n_before, model.n)


def test_morris_lecar_current_forward(model_params):
    model = MorrisLecarCurrent(**model_params)
    v_before = model.mem.clone()
    s_before = model.s.clone()
    n_before = model.n.clone()
    input = torch.zeros(size=model.mem.size(), dtype=torch.float32, device=model_params["device"])
    model.forward(input)

    assert not torch.allclose(v_before, model.mem)
    assert not torch.allclose(s_before, model.s)
    assert not torch.allclose(n_before, model.n)


def test_lif_forward(model_params):
    model = LIF(**model_params)
    v_before = model.mem.clone()
    r_before = model.r.clone()
    input = torch.zeros(size=model.mem.size(), dtype=torch.float32, device=model_params["device"])
    model.forward(input)

    assert not torch.allclose(v_before, model.mem)
    assert not torch.allclose(r_before, model.r)
