"""Morris-Lecar network models."""

from typing import ClassVar

import numpy as np
import torch
from torch import nn

from .base import SNNBase


class MorrisLecar(SNNBase):
    r"""Network of Morris-Lecar neurons with conductance-based synapses.

    The model updates all :math:`N` neurons at once with tensor operations and forward Euler steps.
    See the Notes section for the equations.

    Parameters
    ----------
    dt : float
        Integration time step in ms.
    BIAS : np.ndarray, torch.Tensor, or float
        Bias current :math:`I_{bias}` in pA. Pass one value for all neurons or an array of shape (N, 1).
    Ne : int, default 20
        Number of excitatory neurons. Their synapses use the reversal potential `E_AMPA`.
    Ni : int, default 20
        Number of inhibitory neurons. Their synapses use the reversal potential `E_GABA`.
    C : float, default 20
        Membrane capacitance :math:`C`. It divides the total current, so it also sets the time scale of :math:`v`.
    g_L : float, default 2
        Leak conductance :math:`g_L`.
    g_K : float, default 8
        Potassium conductance :math:`g_K`.
    g_Ca : float, default 4.4
        Calcium conductance :math:`g_{Ca}`.
    E_L : float, default -60
        Reversal potential of the leak current in mV.
    E_K : float, default -84
        Reversal potential of the potassium current in mV.
    E_Ca : float, default 120
        Reversal potential of the calcium current in mV.
    v1 : float, default -1.2
        Midpoint :math:`v_1` of :math:`m_\infty` in mV.
    v2 : float, default 18
        Slope :math:`v_2` of :math:`m_\infty` in mV.
    v3 : float, default 2
        Midpoint :math:`v_3` of :math:`n_\infty` and :math:`\tau_n` in mV.
    v4 : float, default 30
        Slope :math:`v_4` of :math:`n_\infty` and :math:`\tau_n` in mV.
    phi : float, default 0.04
        Rate factor :math:`\phi` of the potassium gate.
    a_r : float, default 1.1
        Rise rate :math:`\alpha_r` of the synaptic gate.
    a_d : float, default 0.02
        Decay rate :math:`\alpha_d` of the synaptic gate.
    v_t : float, default 2
        Midpoint :math:`v_t` of the transmitter release :math:`T(v)` in mV.
    k_p : float, default 5
        Slope :math:`k_p` of :math:`T(v)` in mV.
    t_max : float, default 1.0
        Maximum :math:`T_{max}` of :math:`T(v)`.
    E_AMPA : float, default 0
        Reversal potential of excitatory synapses in mV.
    E_GABA : float, default -75
        Reversal potential of inhibitory synapses in mV.
    gbar : float, default 1
        Synaptic conductance :math:`\bar{g}` in nS.
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
    n : torch.Tensor of shape (N, 1)
        Potassium gating variable. It starts uniformly random in [0, 1).
    s : torch.Tensor of shape (N, 1)
        Synaptic gating variable, which the reservoir reads out. It starts at zero.
    w : torch.Tensor of shape (N, N)
        Boolean connectivity :math:`w`. Each entry is True with probability 2 / N.

    Notes
    -----
    The membrane potential follows

    .. math::

        C \dot{v} = I_{bias} + I_L + I_K + I_{Ca} + I_{in} + I_{syn}

    with the ionic currents

    .. math::

        I_L &= -g_L (v - E_L) \\
        I_K &= -g_K n (v - E_K) \\
        I_{Ca} &= -g_{Ca} m_\infty(v) (v - E_{Ca})

    The gating variables follow

    .. math::

        \dot{n} &= \frac{n_\infty(v) - n}{\tau_n(v)} \\
        \dot{s} &= \alpha_r T(v) (1 - s) - \alpha_d s

    where

    .. math::

        m_\infty(v) &= \frac{1}{2} \left[1 + \tanh\left(\frac{v - v_1}{v_2}\right)\right] \\
        n_\infty(v) &= \frac{1}{2} \left[1 + \tanh\left(\frac{v - v_3}{v_4}\right)\right] \\
        \tau_n(v) &= \frac{1}{\phi \cosh\left(\frac{v - v_3}{2 v_4}\right)} \\
        T(v) &= \frac{T_{max}}{1 + \exp\left(-\frac{v - v_t}{k_p}\right)}

    The synaptic current into neuron :math:`i` is

    .. math::

        I_{syn,i} = -\bar{g} \sum_{j=1}^{N} w_{ij} (v_i - E_j) s_j

    where :math:`E_j` is `E_AMPA` if neuron :math:`j` is excitatory and `E_GABA` if it is inhibitory.
    All outgoing synapses of a neuron share its reversal potential, so the network follows Dale's law.
    """

    __constants__: ClassVar[list[str]] = [
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
        dt: float,
        BIAS: np.ndarray | torch.Tensor | float,
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
        gbar: float = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Morris Lecar model parameters
        self.factory_kwargs = nn.factory_kwargs({"device": device, "dtype": dtype})
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
        """Reset `mem` and `s` to zero and draw a new random `n`."""
        self.mem = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.s = torch.zeros(size=(self.N, 1), **self.factory_kwargs)
        self.n = torch.rand(self.N, 1, **self.factory_kwargs)

    def m_ss(self) -> torch.Tensor:
        r"""Return the steady-state calcium activation :math:`m_\infty(v)`."""
        return 0.5 * (1 + torch.tanh((self.mem - self._v1) / self._v2))

    def n_ss(self) -> torch.Tensor:
        r"""Return the steady-state potassium activation :math:`n_\infty(v)`."""
        return 0.5 * (1 + torch.tanh((self.mem - self._v3) / self._v4))

    def tau_n(self) -> torch.Tensor:
        r"""Return the potassium time constant :math:`\tau_n(v)`."""
        cosh = torch.cosh((self.mem - self._v3) / (2 * self._v4))
        return 1 / (self._phi * cosh)

    def T(self):
        """Return the transmitter release :math:`T(v)`."""
        exp = torch.exp(-(self.mem - self._v_t) / self._k_p)
        return self._t_max / (1 + exp)

    def s_dot(self):
        r"""Return the time derivative :math:`\dot{s}` of the synaptic gate."""
        return self._a_r * self.T() * (1 - self.s) - self._a_d * self.s

    def n_dot(self):
        r"""Return the time derivative :math:`\dot{n}` of the potassium gate."""
        return (self.n_ss() - self.n) / self.tau_n()

    def calc_ipsc(self) -> torch.Tensor:
        r"""Return the synaptic current :math:`I_{syn}` into each neuron, as a tensor of shape (N, 1)."""
        return -self.gbar * self.w * ((self.mem @ self.ones) - self.E) @ self.s

    def check_nan(self, inp: torch.Tensor, name: str):
        """Raise an error if a tensor contains NaN.

        Parameters
        ----------
        inp : torch.Tensor
            Tensor to check.
        name : str
            Name of the tensor for the error message.

        Raises
        ------
        ValueError
            If `inp` contains NaN.
        """
        if inp.isnan().any():
            raise ValueError("Found NaN value in " + name)

    def mem_dot(self, input_: torch.Tensor) -> torch.Tensor:
        r"""Return the time derivative of the membrane potential.

        Parameters
        ----------
        input_ : torch.Tensor of shape (N, 1) or float
            Input current :math:`I_{in}`.

        Returns
        -------
        torch.Tensor of shape (N, 1)
            :math:`\dot{v} = (I_{bias} + I_L + I_K + I_{Ca} + I_{in} + I_{syn}) / C`.
        """
        ps_current = self.calc_ipsc()  # Calculate the new post-synaptic potential
        I_L = -self._g_L * (self.mem - self._v_L)
        I_K = -self._g_K * self.n * (self.mem - self._v_K)
        I_Ca = -self._g_Ca * self.m_ss() * (self.mem - self._v_Ca)

        return (self.BIAS + I_L + I_K + I_Ca + input_ + ps_current) / self.C

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Advance the network by one time step.

        Parameters
        ----------
        input_ : torch.Tensor of shape (N, 1) or float
            Input current for this step.

        Returns
        -------
        torch.Tensor of shape (N, 1)
            The updated `s`. This is the model's own tensor, not a copy.

        Raises
        ------
        ValueError
            If the membrane potential contains NaN after the step.
        """
        dv = self.dt * self.mem_dot(input_)
        self.n += self.dt * self.n_dot()
        self.s += self.dt * self.s_dot()
        self.mem += dv
        self.check_nan(self.mem, "mem")

        return self.s

    def state(self):
        """Return a copy of the synaptic gate `s`."""
        return self.s.clone()


class MorrisLecarCurrent(MorrisLecar):
    r"""Morris-Lecar network with current-based synapses and normally distributed weights.

    The neurons follow the same dynamics as in :class:`MorrisLecar`, but the synaptic current
    into neuron :math:`i` has no reversal potentials:

    .. math::

        I_{syn,i} = -\bar{g} \sum_{j=1}^{N} w_{ij} s_j

    where :math:`w_{ij}` is the weight from neuron :math:`j` to neuron :math:`i`. The sign of each weight,
    not the neuron type, decides whether a synapse excites or inhibits. A neuron can have outgoing synapses
    of both signs, so the network does not follow Dale's law. For other connectivity, assign your own
    weights to `w`.

    The model takes the same parameters as :class:`MorrisLecar`, plus `p_sparsity`. It has no `dtype`
    parameter and always uses float32. `Ne` and `Ni` only set `N`.

    Parameters
    ----------
    p_sparsity : float, default 1.0
        Sets the standard deviation of the weights to :math:`1 / \sqrt{N p}`, where :math:`p` is `p_sparsity`.
        The weights stay dense.

    Attributes
    ----------
    w : torch.Tensor of shape (N, N)
        Recurrent weights :math:`w`, drawn from a normal distribution with mean 0 and standard deviation
        :math:`1 / \sqrt{N p}`.
    """

    def __init__(
        self,
        dt: float,
        BIAS: np.ndarray | torch.Tensor | float,
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
        gbar: float = 1,
        device: torch.device = torch.device("cpu"),
        p_sparsity: float = 1.0,
    ) -> None:
        # Morris Lecar model parameters
        super().__init__(
            dt=dt,
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
            gbar=gbar,
            device=device,
        )
        # Network connections -- Normal Distribution
        self.w = torch.normal(mean=0, std=1 / np.sqrt(self.N * p_sparsity), size=(self.N, self.N), **self.factory_kwargs)

    def calc_ipsc(self) -> torch.Tensor:
        r"""Return the current-based synaptic current :math:`I_{syn} = -\bar{g} w s`, of shape (N, 1)."""
        return -self.gbar * self.w @ self.s
