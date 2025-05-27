from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .base import SNNBase


class LIF(SNNBase):
    def __init__(
        self,
        N: int,
        T,
        dt,
        BIAS: Union[torch.Tensor, np.ndarray, float] = 80.0,
        tau_m: float = 50,
        tau_s: float = 20,
        gbar: float = 5,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.factory_kwargs = nn.factory_kwargs({"device": device, "dtype": dtype})
        self.T = T
        self.dt = dt
        self.time = torch.arange(0, T, dt, **self.factory_kwargs)
        self.nt = self.time.size()[0]
        self.N = N
        self.device = device
        self.BIAS = BIAS  # pA

        self.tau_m = tau_m  # ms
        self.tau_s = tau_s  # ms
        self.v_th = 30  # mV
        self.v_reset = -20  # mV

        self.v = torch.zeros(size=(N, 1), **self.factory_kwargs)
        self.r = torch.zeros(size=(N, 1), **self.factory_kwargs)

        self.gbar = gbar
        self.w = torch.tensor(np.random.normal(loc=0, scale=1 / N, size=(N, N)), **self.factory_kwargs)

    def r_dot(self):
        return -self.r / self.tau_s

    def v_dot(self, input_: torch.Tensor) -> torch.Tensor:
        return (-self.v + self.BIAS + self.gbar * self.w @ self.r + input_) / self.tau_m

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        dv = self.dt * self.v_dot(input_)
        self.r += self.dt * self.r_dot()
        self.v += dv
        mask = self.v > self.v_th
        self.v[mask] = self.v_reset
        self.r[mask] += 1 / self.tau_s

        return self.r
