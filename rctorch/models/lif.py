"""Leaky integrate-and-fire network models."""

import numpy as np
import torch
from torch import nn

from .base import SNNBase


class LIF(SNNBase):
    r"""Network of leaky integrate-and-fire neurons with exponential synapses.

    The membrane potential :math:`v` and synaptic output :math:`r` of the neurons follow

    .. math::

        \tau_m \dot{v} &= -v + I_{bias} + \bar{g} W r + I_{in} \\
        \tau_s \dot{r} &= -r

    When :math:`v` of a neuron crosses :math:`v_{th} = 30` mV, it resets to :math:`v_{reset} = -20` mV
    and :math:`r` of that neuron grows by :math:`1 / \tau_s`. The model integrates both equations with
    forward Euler steps.

    Parameters
    ----------
    Ne : int
        Number of excitatory neurons.
    Ni : int
        Number of inhibitory neurons.
    dt : float
        Integration time step in ms.
    BIAS : torch.Tensor, np.ndarray, or float, default 80.0
        Bias current :math:`I_{bias}` in pA. Pass one value for all neurons or an array of shape (N, 1).
    tau_m : float, default 50
        Membrane time constant :math:`\tau_m` in ms.
    tau_s : float, default 20
        Synaptic time constant :math:`\tau_s` in ms.
    gbar : float, default 5
        Coupling strength :math:`\bar{g}`.
    device : torch.device, default torch.device("cpu")
        Device for the model's tensors.
    dtype : torch.dtype, default torch.float32
        Data type for the model's tensors.

    Attributes
    ----------
    N : int
        Total number of neurons, ``Ne + Ni``.
    mem : torch.Tensor of shape (N, 1)
        Membrane potential :math:`v` in mV. It starts at zero.
    r : torch.Tensor of shape (N, 1)
        Synaptic output, which the reservoir reads out. It starts uniformly random in [0, 1).
    w : torch.Tensor of shape (N, N)
        Recurrent weights :math:`W`, drawn from a normal distribution with mean 0 and standard deviation 1 / N.

    Notes
    -----
    `Ne` and `Ni` only set `N`. The weights do not depend on whether a neuron is excitatory or inhibitory.
    """

    def __init__(
        self,
        Ne: int,
        Ni: int,
        dt,
        BIAS: torch.Tensor | np.ndarray | float = 80.0,
        tau_m: float = 50,
        tau_s: float = 20,
        gbar: float = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_hidden=Ne + Ni, dt=dt, dtype=dtype, device=device)
        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni
        self.BIAS = BIAS  # pA

        self.tau_m = tau_m  # ms
        self.tau_s = tau_s  # ms
        self.v_th = 30  # mV
        self.v_reset = -20  # mV

        self.mem = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.r = torch.rand(size=(self.N, 1), **self.factory_kwargs)

        self.gbar = gbar
        self.w = torch.tensor(np.random.normal(loc=0, scale=1 / self.N, size=(self.N, self.N)), **self.factory_kwargs)

    def r_dot(self):
        r"""Return the time derivative of the synaptic output, :math:`\dot{r} = -r / \tau_s`."""
        return -self.r / self.tau_s

    def mem_dot(self, input_: torch.Tensor) -> torch.Tensor:
        r"""Return the time derivative of the membrane potential.

        Parameters
        ----------
        input_ : torch.Tensor of shape (N, 1) or float
            Input current :math:`I_{in}`.

        Returns
        -------
        torch.Tensor of shape (N, 1)
            :math:`\dot{v} = (-v + I_{bias} + \bar{g} W r + I_{in}) / \tau_m`.
        """
        return (-self.mem + self.BIAS + self.gbar * self.w @ self.r + input_) / self.tau_m

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Advance the network by one time step, then reset the neurons that spiked.

        Parameters
        ----------
        input_ : torch.Tensor of shape (N, 1) or float
            Input current for this step.

        Returns
        -------
        torch.Tensor of shape (N, 1)
            The updated `r`. This is the model's own tensor, not a copy.
        """
        dv = self.dt * self.mem_dot(input_)
        self.r += self.dt * self.r_dot()
        self.mem += dv
        mask = self.mem > self.v_th
        self.mem[mask] = self.v_reset
        self.r[mask] += 1 / self.tau_s

        return self.r

    def state(self) -> torch.Tensor:
        """Return a copy of the synaptic output `r`."""
        return self.r.clone()


class LIFDoubleExp(LIF):
    def __init__(
        self,
        Ne: int,
        Ni: int,
        dt,
        BIAS: torch.Tensor | np.ndarray | float = 80.0,
        tau_m: float = 50,
        tau_r: float = 2,
        tau_d: float = 20,
        gbar: float = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.factory_kwargs = nn.factory_kwargs({"device": device, "dtype": dtype})
        self.dt = dt
        self.Ne = Ne
        self.Ni = Ni
        self.N = Ne + Ni
        self.BIAS = BIAS  # pA

        self.tau_m = tau_m  # ms
        self.tau_r = tau_r  # ms
        self.tau_d = tau_d  # ms
        self.v_th = 30  # mV
        self.v_reset = -20  # mV

        self.mem = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.r = torch.rand(size=(self.N, 1), **self.factory_kwargs)

        self.gbar = gbar
        self.w = torch.tensor(np.random.normal(loc=0, scale=1 / self.N, size=(self.N, self.N)), **self.factory_kwargs)

    # TODO: Implement double exponential
    def r_dot(self):
        return -self.r / self.tau_s

    def mem_dot(self, input_: torch.Tensor) -> torch.Tensor:
        return (-self.mem + self.BIAS + self.gbar * self.w @ self.r + input_) / self.tau_m

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        dv = self.dt * self.mem_dot(input_)
        self.r += self.dt * self.r_dot()
        self.mem += dv
        mask = self.mem > self.v_th
        self.mem[mask] = self.v_reset
        self.r[mask] += 1 / self.tau_s

        return self.r

    def state(self) -> torch.Tensor:
        return self.r.clone()
