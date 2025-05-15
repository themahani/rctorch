import numpy as np
import torch
from tqdm import tqdm


class LIF:
    def __init__(
        self,
        device: torch.device,
        N: int,
        T,
        dt,
        sup: np.ndarray,
        tau_m: float = 50,
        tau_s: float = 20,
        BIAS: float = 80,
        G: float = 5,
        Q: float = 200,
        ridge_coeff: float = 1.0,
    ) -> None:
        self.T = T
        self.dt = dt
        self.time = torch.arange(0, T, dt, device=device)
        self.nt = self.time.size()[0]
        self.N = N
        self.device = device
        self._BIAS = BIAS  # pA

        self.tau_m = tau_m  # ms
        self.tau_s = tau_s  # ms
        self.v_th = 30  # mV
        self.v_reset = -20  # mV

        self.v = torch.zeros(size=(N, 1), dtype=torch.float32, device=device)
        self.r = torch.zeros(size=(N, 1), dtype=torch.float32, device=device)

        self.G = G
        self.w = self.G * torch.tensor(
            np.random.normal(loc=0, scale=1 / N, size=(N, N)),
            dtype=torch.float32,
            device=device,
        )

        self.Q = Q
        self.ridge_coeff = ridge_coeff
        self.dim = min(sup.shape)
        self.enc = Q * (2 * torch.rand(N, self.dim, dtype=torch.float32, device=self.device) - 1)
        self.dec = torch.zeros(size=(N, self.dim), dtype=torch.float32, device=device)
        self.Pinv = torch.eye(N, dtype=torch.float32, device=device) / ridge_coeff
        self.x_hat = self.dec.T @ self.r
        self.x_hat_rec = torch.zeros(size=(self.nt, self.x_hat.size()[0]), device=device)
        self.sup = torch.tensor(sup, dtype=torch.float32, device=device)

    def __reinit__(self):
        self.v = torch.zeros(size=(self.N, 1), dtype=torch.float32, device=self.device)
        self.r = torch.zeros(size=(self.N, 1), dtype=torch.float32, device=self.device)
        self.x_hat_rec = torch.zeros(size=(self.nt, self.x_hat.size()[0]), device=self.device)

    def r_dot(self):
        return -self.r / self.tau_s

    def v_dot(self):
        return (-self.v + self.I + self.w @ self.r + self.enc @ self.x_hat) / self.tau_m

    def calculate_x_hat(self):
        self.x_hat = self.dec.T @ self.r

    def rls(self, i):
        error = self.x_hat - self.sup[i].reshape(-1, 1)
        q = self.Pinv @ self.r
        self.Pinv -= (q @ q.T) / (1 + self.r.T @ q)
        self.dec -= (self.Pinv @ self.r) @ error.T

    def _euler_step(self):
        dv = self.dt * self.v_dot()
        self.r += self.dt * self.r_dot()
        self.v += dv
        mask = self.v > self.v_th
        self.v[mask] = self.v_reset
        self.r[mask] += 1 / self.tau_s
        self.x_hat = self.dec.T @ self.r

    def render(self, rls_start, rls_stop, rls_step):
        rls_stop = round(rls_stop // self.dt)
        rls_start = round(rls_start // self.dt)

        for i in tqdm(range(self.nt)):
            self._euler_step()

            self.x_hat_rec[i, :] = self.x_hat.flatten()

            if i > rls_start and i < rls_stop:
                if i % rls_step == 1:
                    self.rls(i)

    def _train(self, rls_stop: float, rls_step: int, transient_time: float = 200):
        rls_stop = round(rls_stop // self.dt)
        nt = round(transient_time // self.dt)

        print("Transient time...")
        for i in tqdm(range(nt)):
            self._euler_step()

        print("Training time...")
        for i in tqdm(range(rls_stop)):
            self._euler_step()

            self.x_hat_rec[i, :] = self.x_hat.flatten()

            if i % rls_step == 1:
                self.rls(i)

    def epoch_train(self, rls_stop: float, rls_step: int, n_epochs=10):
        """Train the model for multiple epochs

        Parameters
        ----------
        rls_stop : float
            When to stop the training in [ms]
        rls_step : int
            The timestep interval between training using RLS
        n_epochs : int, optional
            Number of epochs to train the model, by default 10
        """
        for j in range(n_epochs):
            print(f"Training epoch {j+1}/{n_epochs}")
            self.__reinit()
            self._train(rls_stop, rls_step)
