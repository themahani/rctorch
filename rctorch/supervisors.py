"""Generators for the target signals that a reservoir learns to reproduce."""

import numpy as np


class VanDerPol:
    r"""Van der Pol oscillator.

    .. math::

        \dot{x} &= \tau y \\
        \dot{y} &= \tau \left[\mu (1 - x^2) y - x\right]

    The generator integrates the system with forward Euler steps and updates :math:`x` before :math:`y`.

    Parameters
    ----------
    T : float
        Duration of the generated signal, in the same unit as `dt`.
    dt : float
        Time step.
    mu : float
        Nonlinearity and damping strength :math:`\mu`.
    tau : float
        Time-scale factor :math:`\tau`. Larger values make the oscillation faster.
    x0 : float, default 1
        Initial value of :math:`x`.
    y0 : float, default 1
        Initial value of :math:`y`.

    Attributes
    ----------
    nt : int
        Number of recorded steps, ``round(T / dt)``.
    data : np.ndarray of shape (2, nt)
        The signal from the last call to :meth:`generate`.
    """

    def __init__(
        self, T: float, dt: float, mu: float, tau: float, x0: float = 1, y0: float = 1
    ) -> None:
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
        """Integrate the oscillator and return the trajectory.

        The state carries over between calls, so a second call continues where the first one stopped.

        Parameters
        ----------
        transient_time : float, default 200
            Time to integrate and discard before recording, in the same unit as `dt`.

        Returns
        -------
        np.ndarray of shape (2, nt)
            Rows are :math:`x` and :math:`y`.
        """
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
    r"""Lorenz system.

    .. math::

        \dot{x} &= \tau \sigma (y - x) \\
        \dot{y} &= \tau \left[x (\rho - z) - y\right] \\
        \dot{z} &= \tau (x y - \beta z)

    The generator integrates the system with forward Euler steps.

    Parameters
    ----------
    T : float
        Duration of the generated signal, in the same unit as `dt`.
    dt : float
        Time step.
    tau : float
        Time-scale factor :math:`\tau`. Larger values make the trajectory faster.
    sigma : float, default 10.0
        Parameter :math:`\sigma`.
    rho : float, default 28.0
        Parameter :math:`\rho`.
    beta : float, default 8/3
        Parameter :math:`\beta`.
    init : np.ndarray of shape (3,), optional
        Initial state :math:`(x, y, z)`. Defaults to ``[1.0, 0.0, 0.5]``. The generator copies the array
        and leaves the caller's array unchanged.

    Attributes
    ----------
    nt : int
        Number of recorded steps, ``round(T / dt)``.
    state : np.ndarray of shape (3,)
        Current state :math:`(x, y, z)`.
    data : np.ndarray of shape (3, nt)
        The signal from the last call to :meth:`generate`.
    """

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
        """Integrate the system and return the trajectory.

        The state carries over between calls, so a second call continues where the first one stopped.

        Parameters
        ----------
        transient_time : float, default 0.0
            Time to integrate and discard before recording, in the same unit as `dt`.

        Returns
        -------
        np.ndarray of shape (3, nt)
            Rows are :math:`x`, :math:`y`, and :math:`z`.
        """
        if transient_time > 0.0:
            trans_nt = round(transient_time / self.dt)
            for i in range(trans_nt):
                self.state += self.dt * self._x_dot() * self.tau

        for i in range(self.nt):
            self.state += self.dt * self._x_dot() * self.tau
            self.data[:, i] = self.state

        return self.data


class HyperChaoticAttractor:
    r"""Rössler hyperchaotic system.

    .. math::

        \dot{x}_1 &= \tau (-x_2 - x_3) \\
        \dot{x}_2 &= \tau (x_1 + a x_2 + x_4) \\
        \dot{x}_3 &= \tau (b + x_1 x_3) \\
        \dot{x}_4 &= \tau (-c x_3 + d x_4)

    with :math:`a = 0.25`, :math:`b = 3`, :math:`c = 0.5`, and :math:`d = 0.05`. The state starts at
    :math:`(-10, -6, 0, 10)`, and the generator integrates the system with forward Euler steps.

    Parameters
    ----------
    T : float
        Duration of the generated signal, in the same unit as `dt`.
    dt : float
        Time step.
    tau : float
        Time-scale factor :math:`\tau`. Larger values make the trajectory faster.

    Attributes
    ----------
    nt : int
        Number of recorded steps, ``round(T / dt)``.
    state : np.ndarray of shape (4,)
        Current state :math:`(x_1, x_2, x_3, x_4)`.
    data : np.ndarray of shape (4, nt)
        The signal from the last call to :meth:`generate`.
    """

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
        """Integrate the system and return the trajectory.

        The state carries over between calls, so a second call continues where the first one stopped.

        Returns
        -------
        np.ndarray of shape (4, nt)
            Rows are :math:`x_1` through :math:`x_4`.
        """
        for i in range(self.nt):
            self.state += self.dt * self._x_dot() * self.tau
            self.data[:, i] = self.state

        return self.data
