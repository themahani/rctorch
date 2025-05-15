import pytest

from ml_force import HyperChaoticAttractor, LorenzAttractor, VanDerPol


@pytest.fixture
def time_params():
    return {"T": 1000.0, "dt": 0.1, "tau": 0.01}


def test_van_der_pol(time_params):
    vdp = VanDerPol(**time_params, mu=1.0)
    data = vdp.generate()
    assert data.shape[0] == 2
    assert data.shape[1] == round(time_params["T"] / time_params["dt"])


def test_lorenz_attractor(time_params):
    la = LorenzAttractor(**time_params)
    data = la.generate()
    assert data.shape[0] == 3
    assert data.shape[1] == round(time_params["T"] / time_params["dt"])


def test_hyper_chaotic_attractor(time_params):
    hca = HyperChaoticAttractor(**time_params)
    data = hca.generate()
    assert data.shape[0] == 4
    assert data.shape[1] == round(time_params["T"] / time_params["dt"])


def test_van_der_pol_transient(time_params):
    vdp = VanDerPol(**time_params, mu=1.0)
    trans_time = 200
    data = vdp.generate(transient_time=trans_time)
    assert data.shape[0] == 2
    assert data.shape[1] == round(time_params["T"] / time_params["dt"])


def test_lorenz_attractor_params(time_params):
    params = {"sigma": 15.0, "rho": 35.0, "beta": 3.0}
    la = LorenzAttractor(**time_params, **params)
    assert la.sigma == params["sigma"]
    assert la.rho == params["rho"]
    assert la.beta == params["beta"]
