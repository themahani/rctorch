#!/usr/bin/env python

import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import factory_kwargs
from tqdm import tqdm

from .base import SNNBase


class MorrisLecar(SNNBase):
    __constants__ = [
        "dt",
        "Ne",
        "Ni",
        "C",
        "_g_L",
        "_g_Ca",
        "_g_K",
        "_v_L",
        "_v_Ca",
        "_v_K",
        "_v1",
        "_v2",
        "_v3",
        "_v4",
        "_phi",
        "_a_r",
        "_a_d",
        "_v_t",
        "_k_p",
        "_t_max",
        "_v_AMPA",
        "_v_GABA",
    ]

    def __init__(
        self,
        supervisor: np.ndarray,
        dt: float,
        T: float,
        BIAS: Union[np.ndarray, Tensor, np.float64],
        Ne: int = 20,
        Ni: int = 20,
        C: float = 20,
        g_L: float = 2,
        g_K: float = 8,
        g_Ca: float = 4.4,
        E_L: float = -60,
        E_K: float = -84,
        E_Ca: float = 120,
        v1: float = -1.2,
        v2: float = 18,
        v3: float = 2,
        v4: float = 30,
        phi: float = 0.04,
        a_r: float = 1.1,
        a_d: float = 0.02,
        v_t: float = 2,
        k_p: float = 5,
        t_max: float = 1.0,
        E_AMPA: float = 0,
        E_GABA: float = -75,
        Q: float = 100,
        ridge_coeff: float = 1.0,
        gbar: float = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """A modified version of the Morris Lecar neural network model. This model uses a block structure to
        simplify and optimize for matrix multiplications using `Numpy`.

        Parameters
        ----------
        supervisor : np.ndarray
            The target output of the model. This is the signal the model is supposed to follow after being trained
        dt : float
            Time step for the neural networks and the supervisor in [ms].
        T : float
            The total simulation time of the system.
        BIAS : np.ndarray
            The input bias into all the neurons in the reservoir in [pA].
            Can be engineered.
        Ne : int, optional
            The number of excitatory neurons in the reservoir, by default 20
        Ni : int, optional
            The number of inhibitory neurons in the reservoir, by default 20
        C : float, optional
            The capaticance of the Morris Lecar model. Also acts as the time constant of the nerwork, by default 20
        g_L : float, optional
            Leak conductance, by default 2
        g_K : float, optional
            Potassium conductance, by default 8
        g_Ca : float, optional
            Calsium conductance, by default 4.4
        E_L : float, optional
            Baseline voltage for Leak, by default -60
        E_K : float, optional
            Baseline voltage for K, by default -84
        E_Ca : float, optional
            Baseline voltage for Calsium, by default 120
        v1 : float, optional
            Voltage param for the model dynamics, by default -1.2
        v2 : float, optional
            Voltage param for the model dynamics, by default 18
        v3 : float, optional
            Voltage param for the model dynamics, by default 2
        v4 : float, optional
            Voltage param for the model dynamics, by default 30
        phi : float, optional
            Frequency in `n_dot`, by default 0.04
        a_r : float, optional
            Synaptic rise time constant, by default .2
        a_d : float, optional
            Synaptic decay time constant, by default 0.02
        v_t : float, optional
            Constant for `T`, by default 2
        k_p : float, optional
            Constant for `T`, by default 5
        t_max : float, optional
            Constant for `T`, by default 1.0
        E_AMPA : float, optional
            The excitatory resting potential in [mV], by default 0
        E_GABA : float, optional
            The inhibitory resting potential in [mV], by default -75
        Q : float, optional
            The encoding strength coefficient, by default 2
        ridge_coeff : float, optional
            The learning strength for FORCE training, by default 1.0
        gbar : float, optional
            Synaptic conductance for all neurons in [nS], by default 1
        """
        # Morris Lecar model parameters
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni
        self.BIAS = torch.tensor(BIAS, **self.factory_kwargs)
        self.C = C
        self.gbar = gbar
        self.dt = dt
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._v_L = E_L
        self._v_K = E_K
        self._v_Ca = E_Ca
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._phi = phi
        self._v_t = v_t
        self._a_r = a_r
        self._a_d = a_d
        self._k_p = k_p
        self._t_max = t_max
        self._v_AMPA = E_AMPA
        self._v_GABA = E_GABA

        # Network connections
        self.mem = torch.zeros(size=(self.N, 1), **self.factory_kwargs)  # Membrane potential
        self.s = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.n = torch.rand(self.N, 1, **self.factory_kwargs)
        self.w = torch.rand(self.N, self.N, device=self.device) < (2 / self.N)
        self.E = torch.tensor([E_AMPA] * Ne + [E_GABA] * Ni, **self.factory_kwargs).reshape(self.N, 1)
        # Let's follow Dale's law
        self.ones = torch.ones(size=(1, self.N), **self.factory_kwargs)
        self.E = (self.E @ self.ones).T

    def __reinit__(self):
        self.mem = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.s = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.n = torch.rand(self.N, 1, **self.factory_kwargs)

    def m_ss(self) -> Tensor:
        return 0.5 * (1 + torch.tanh((self.mem - self._v1) / self._v2))

    def n_ss(self) -> Tensor:
        return 0.5 * (1 + torch.tanh((self.mem - self._v3) / self._v4))

    def tau_n(self) -> Tensor:
        cosh = torch.cosh((self.mem - self._v3) / (2 * self._v4))
        return 1 / (self._phi * cosh)

    def T(self):
        exp = torch.exp(-(self.mem - self._v_t) / self._k_p)
        return self._t_max / (1 + exp)

    def s_dot(self):
        return self._a_r * self.T() * (1 - self.s) - self._a_d * self.s

    def n_dot(self):
        return (self.n_ss() - self.n) / self.tau_n()

    def calc_ipsc(self) -> Tensor:
        return -self.gbar * self.w * ((self.mem @ self.ones) - self.E) @ self.s

    def check_nan(self, inp: Tensor, name: str):
        if inp.isnan().any():
            raise ValueError("Found NaN value in " + name)

    def mem_dot(self, input_: Tensor) -> Tensor:
        self.calc_ipsc()  # Calculate the new post-synaptic potential
        I_L = -self._g_L * (self.mem - self._v_L)
        I_K = -self._g_K * self.n * (self.mem - self._v_K)
        I_Ca = -self._g_Ca * self.m_ss() * (self.mem - self._v_Ca)

        return (self.BIAS + I_L + I_K + I_Ca + input_) / self._C

    def forward(self, input_: Tensor) -> None:
        dv = self._dt * self.mem_dot(input_)
        self.n += self._dt * self.n_dot()
        self.s += self._dt * self.s_dot()
        self.mem += dv
        self.check_nan(self.mem, "mem")


class MorrisLecarCurrent(MorrisLecar):
    def __init__(
        self,
        supervisor: np.ndarray,
        dt: float,
        T: float,
        BIAS: Union[np.ndarray, np.float64],
        Ne: int = 20,
        Ni: int = 20,
        C: float = 20,
        g_L: float = 2,
        g_K: float = 8,
        g_Ca: float = 4.4,
        E_L: float = -60,
        E_K: float = -84,
        E_Ca: float = 120,
        v1: float = -1.2,
        v2: float = 18,
        v3: float = 2,
        v4: float = 30,
        phi: float = 0.04,
        a_r: float = 1.1,
        a_d: float = 0.02,
        v_t: float = 2,
        k_p: float = 5,
        t_max: float = 1.0,
        E_AMPA: float = 0,
        E_GABA: float = -75,
        Q: float = 100,
        ridge_coeff: float = 1.0,
        gbar: float = 1,
        w_rand: float = 0.0,
        device: torch.device = torch.device("cpu"),
        p_sparsity: float = 1.0,
    ) -> None:
        """A modified version of the Morris Lecar neural network model. This model uses a block structure to
        simplify and optimize for matrix multiplications using `PyTorch`.

        Parameters
        ----------
        supervisor : np.ndarray
            The target output of the model. This is the signal the model is supposed to follow after being trained
        dt : float
            Time step for the neural networks and the supervisor in [ms].
        T : float
            The total simulation time of the system.
        BIAS : np.ndarray
            The input bias into all the neurons in the reservoir in [pA].
            Can be engineered.
        Ne : int, optional
            The number of excitatory neurons in the reservoir, by default 20
        Ni : int, optional
            The number of inhibitory neurons in the reservoir, by default 20
        C : float, optional
            The capaticance of the Morris Lecar model. Also acts as the time constant of the nerwork, by default 20
        g_L : float, optional
            Leak conductance, by default 2
        g_K : float, optional
            Potassium conductance, by default 8
        g_Ca : float, optional
            Calsium conductance, by default 4.4
        E_L : float, optional
            Baseline voltage for Leak, by default -60
        E_K : float, optional
            Baseline voltage for K, by default -84
        E_Ca : float, optional
            Baseline voltage for Calsium, by default 120
        v1 : float, optional
            Voltage param for the model dynamics, by default -1.2
        v2 : float, optional
            Voltage param for the model dynamics, by default 18
        v3 : float, optional
            Voltage param for the model dynamics, by default 2
        v4 : float, optional
            Voltage param for the model dynamics, by default 30
        phi : float, optional
            Frequency in `n_dot`, by default 0.04
        a_r : float, optional
            Synaptic rise time constant, by default .2
        a_d : float, optional
            Synaptic decay time constant, by default 0.02
        v_t : float, optional
            Constant for `T`, by default 2
        k_p : float, optional
            Constant for `T`, by default 5
        t_max : float, optional
            Constant for `T`, by default 1.0
        E_AMPA : float, optional
            The excitatory resting potential in [mV], by default 0
        E_GABA : float, optional
            The inhibitory resting potential in [mV], by default -75
        Q : float, optional
            The encoding strength coefficient, by default 2
        ridge_coeff : float, optional
            Ridge regression coefficient for the RLS algorithm
        gbar : float, optional
            Synaptic conductance for all neurons in [nS], by default 1
        """
        # Morris Lecar model parameters
        super().__init__(
            supervisor=supervisor,
            dt=dt,
            T=T,
            BIAS=BIAS,
            Ne=Ne,
            Ni=Ni,
            C=C,
            g_L=g_L,
            g_K=g_K,
            g_Ca=g_Ca,
            E_L=E_L,
            E_K=E_K,
            E_Ca=E_Ca,
            v1=v1,
            v2=v2,
            v3=v3,
            v4=v4,
            phi=phi,
            a_r=a_r,
            a_d=a_d,
            v_t=v_t,
            k_p=k_p,
            t_max=t_max,
            E_AMPA=E_AMPA,
            E_GABA=E_GABA,
            Q=Q,
            ridge_coeff=ridge_coeff,
            gbar=gbar,
            device=device,
        )

        # Network connections -- Normal Distribution
        self.w = torch.normal(
            mean=0,
            std=1 / np.sqrt(self.N * p_sparsity),
            size=(self.N, self.N),
            device=self.device,
            dtype=torch.float32,
        )

    def calc_ipsc(self) -> None:
        self.ipsc = -self.gbar * self.w @ self.s
