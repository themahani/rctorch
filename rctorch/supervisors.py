"""This module contains various supervisor generators."""

import numpy as np


class VanDerPol:
    def __init__(self, T: float, dt: float, mu: float, tau: float, x0: float = 1, y0: float = 1) -> None:
        self.T = T
        self.dt = dt
        self.nt = round(self.T / self.dt)
        self.data = np.zeros(shape=(2, self.nt))
        self.mu = mu
        self.tau = tau
        self._x = x0
        self._y = y0

    def _x_dot(self, x, y):
        return y

    def _y_dot(self, x, y):
        return self.mu * (1 - x**2) * y - x

    def generate(self, transient_time: float = 200):
        if transient_time > 0.0:
            trans_nt = round(transient_time / self.dt)
            for i in range(trans_nt):
                self._x += self._x_dot(self._x, self._y) * self.dt * self.tau
                self._y += self._y_dot(self._x, self._y) * self.dt * self.tau

        for i in range(self.nt):
            self._x += self._x_dot(self._x, self._y) * self.dt * self.tau
            self._y += self._y_dot(self._x, self._y) * self.dt * self.tau

            self.data[:, i] = [self._x, self._y]

        return self.data


class LorenzAttractor:
    def __init__(
        self,
        T: float,
        dt: float,
        tau: float,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3.0,
        init: np.ndarray | None = None,
    ) -> None:
        # Time related
        self.T = T
        self.dt = dt
        self.nt = round(self.T / self.dt)
        self.data = np.zeros(shape=(3, self.nt))
        self.state = np.array([1.0, 0.0, 0.5] if init is None else init, dtype=float)
        self.tau = tau  # Integration time constant
        # Parameters of the system
        self.sigma = sigma
        self.beta = beta
        self.rho = rho

    def _x_dot(self) -> np.ndarray:
        gradient = np.zeros(shape=(3,), dtype=float)
        gradient[0] = self.sigma * (self.state[1] - self.state[0])
        gradient[1] = self.state[0] * (self.rho - self.state[2]) - self.state[1]
        gradient[2] = self.state[0] * self.state[1] - self.beta * self.state[2]

        return gradient

    def generate(self, transient_time: float = 0.0):
        if transient_time > 0.0:
            trans_nt = round(transient_time / self.dt)
            for i in range(trans_nt):
                self.state += self.dt * self._x_dot() * self.tau

        for i in range(self.nt):
            self.state += self.dt * self._x_dot() * self.tau
            self.data[:, i] = self.state

        return self.data


class HyperChaoticAttractor:
    def __init__(self, T: float, dt: float, tau: float) -> None:
        # Time related
        self.T = T
        self.dt = dt
        self.nt = round(self.T / self.dt)
        self.data = np.zeros(shape=(4, self.nt))
        self.state = np.array([-10.0, -6.0, 0.0, 10.0])
        self.tau = tau  # Integration time constant
        self.a = 0.25
        self.b = 3.0
        self.c = 0.5
        self.d = 0.05

    def _x_dot(self) -> np.ndarray:
        gradient = np.zeros(shape=(4,), dtype=float)
        gradient[0] = -self.state[1] - self.state[2]
        gradient[1] = self.state[0] + self.a * self.state[1] + self.state[3]
        gradient[2] = self.b + self.state[0] * self.state[2]
        gradient[3] = -self.c * self.state[2] + self.d * self.state[3]

        return gradient

    def generate(self):
        for i in range(self.nt):
            self.state += self.dt * self._x_dot() * self.tau
            self.data[:, i] = self.state

        return self.data
