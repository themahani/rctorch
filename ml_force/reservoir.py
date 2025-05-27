"""
This module contains the implementation of the base `Reservoir`
"""

import torch
from tqdm import tqdm

from .models.base import SNNBase


class Reservoir(torch.nn.Module):
    def __init__(self, model_cls: SNNBase, n_input: int, n_output: int, w_in_amp: float = 200.0, **model_kwargs) -> None:
        super(Reservoir, self).__init__()
        self.model = model_cls(**model_kwargs)
        self.n_hidden = self.model.N
        self.n_input = n_input
        self.n_output = n_output
        # Input and Output layers
        self.W_in = w_in_amp * (2 * torch.rand((self.model.N, self.n_input), **self.model.factory_kwargs) - 1)
        self.W_out = torch.nn.Parameter(
            torch.zeros((self.model.N, self.n_output), **self.model.factory_kwargs), requires_grad=False
        )
        self.Pinv = None

    def forward(self, x: torch.Tensor, nt_transient: int = 500, closed_loop: bool = True):
        """
        Parameters
        ----------
        x: torch.Tensor (nt, n_input)
            Input of the module
        """
        nt = x.size(0)

        if closed_loop and self.n_input != self.n_output:
            raise ValueError("In closed loop model, `n_input` must equal `n_output`.")

        # Record the state of the system
        s_rec = torch.zeros((nt, self.n_hidden), **self.model.factory_kwargs)
        # Transient Period
        print(f"Transient Period:")
        for i in tqdm(range(nt_transient)):
            self.model.forward(0)
        # Prepare the input signal current
        input_current = 0.0
        for i in tqdm(range(nt)):
            s_rec[i] = self.model.state().squeeze()  # Record the SNN state
            if closed_loop:
                x_hat = self.W_out.t() @ s_rec[i].reshape(self.n_hidden, 1)  # Feedback loop
                input_current = self.W_in @ x_hat
            else:
                input_current = self.W_in @ x[i].reshape(self.n_input, 1)
            self.model.forward(input_=input_current)

        return s_rec

    def fit_echo_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        nt_transient: int = 500,
        ridge_reg: float = 1e-5,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor (nt, n_input)
            Input data
        y: torch.Tensor (nt, n_output)
            Output data
        nt_transient: int default 500
            Number of time steps to evolve the reservoir in transient mode (without input)
        ridge_reg: float, default 1e-5
            Ridge regression coefficient
        """
        s_rec = self.forward(x, nt_transient=nt_transient, closed_loop=False)  # forward task for training is open loop
        Pinv = torch.linalg.pinv(s_rec.T @ s_rec + ridge_reg * torch.eye(self.n_hidden, **self.model.factory_kwargs))
        self.W_out.data = Pinv @ s_rec.T @ y
        y_hat = s_rec @ self.W_out

        return y_hat

    def fit_force(
        self, x: torch.Tensor, nt_transient: int = 500, ridge_reg: float = 1.0, rls_step: int = 20, ff_coeff: float = 1.0
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor (nt, n_input)
            Input signal
        nt_transient: int, default 500
            Number of timesteps in transient
        ridge_reg: float, default 1.0
            Ridge regulariztion coefficient for the RLS algorithm
        ff_coeff: float, default 1.0
            Forgetting factor for the RLS algorithm
        """
        if self.n_input != self.n_output:
            raise ValueError("When using the FORCE learning method, `n_input` must equal `n_output`.")
        # Run transient period
        print("Transient Period")
        for _ in tqdm(range(nt_transient)):
            self.model.forward()
        # Run main training loop
        nt = x.size(0)
        x_hat_rec = torch.zeros((nt, self.model.N), **self.model.factory_kwargs)
        if self.Pinv is None:
            self.Pinv = torch.eye(self.model.N, **self.model.factory_kwargs) / ridge_reg

        x_hat = self.W_out.t() @ self.model.state()
        for i in tqdm(range(nt)):
            state = self.model.state()
            x_hat = self.W_out.t() @ state
            x_hat_rec[i] = x_hat.squeeze()
            input_current = self.W_in @ x_hat
            self.model.forward(input_current=input_current)

            if i % rls_step == 0:
                self._rls(x[i].unsqueeze(1), x_hat, state, ff_coeff)
        return x_hat_rec

    def _rls(self, x: torch.Tensor, x_hat: torch.Tensor, state: torch.Tensor, ff_coeff: float = 1.0):
        """
        Parameters
        ----------
        x: torch.Tensor (n_input, 1)
            Input of the model
        x_hat: torch.Tensor (n_input, 1)
            Prediction of the model
        state: torch.Tensor (model.N, 1)
            Current State of the reservoir
        ff_coeff: float, default 1.0
            Forgetting factor for the RLS algorithm, defaults to 1.0 resulting in infinite memory

        Raises
        ------
        ValueError
            If the shapes don't match for the attributes involved.
        TypeError
            If Pinv is not initialized correctly.
        """
        if x.size(0) != self.n_input or x_hat.size(0) != self.n_input:
            raise ValueError("`x_hat`, or `x` should have matching first dimension of `n_input`.")

        if state.size(0) != self.model.N:
            raise ValueError("Model state should have a first dimension of `model.N`")

        if self.Pinv is None:
            raise TypeError("`Pinv` is set to None.")

        u = self.Pinv @ state
        k = u / (ff_coeff + state.t() @ u)
        error = x_hat - x
        self.Pinv = (self.Pinv - k @ (u.T)) / ff_coeff
        self.W_out -= k @ error.T
        return error
