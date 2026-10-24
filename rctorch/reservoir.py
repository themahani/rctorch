"""Reservoir computer built on a spiking neural network."""

import torch
from tqdm import tqdm

from .models.base import SNNBase


class Reservoir(torch.nn.Module):
    r"""Reservoir computer with a spiking network as the hidden layer.

    The input reaches the network through fixed random weights :math:`W_{in}`. The output is a linear
    readout of the network state :math:`r` through trained weights :math:`W_{out}`:

    .. math::

        I_{in} = W_{in} x, \qquad \hat{x} = W_{out}^T r

    Parameters
    ----------
    model_cls : type[SNNBase]
        Spiking network class for the hidden layer, such as :class:`~rctorch.models.LIF`.
    n_input : int
        Dimension of the input signal.
    n_output : int
        Dimension of the output signal.
    w_in_amp : float, default 200.0
        Amplitude of the input weights. Each entry of `W_in` is drawn uniformly from ``[-w_in_amp, w_in_amp]``.
    **model_kwargs
        Keyword arguments for `model_cls`.

    Attributes
    ----------
    model : SNNBase
        The spiking network.
    n_hidden : int
        Number of neurons in `model`.
    W_in : torch.Tensor of shape (n_hidden, n_input)
        Fixed input weights.
    W_out : torch.nn.Parameter of shape (n_hidden, n_output)
        Output weights with ``requires_grad=False``. They start at zero, and :meth:`fit_echo_state` or
        :meth:`fit_force` sets them.
    Pinv : torch.Tensor of shape (n_hidden, n_hidden) or None
        Inverse correlation matrix that :meth:`fit_force` updates. It is None until the first call to
        :meth:`fit_force` and keeps its value between calls.
    """

    def __init__(self, model_cls: SNNBase, n_input: int, n_output: int, w_in_amp: float = 200.0, **model_kwargs) -> None:
        super().__init__()
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
        r"""Run the reservoir and record its state.

        The network first runs for `nt_transient` steps with zero input. It then runs one step per row
        of `x`. In open loop, the input at step ``i`` is ``x[i]``. In closed loop, the input is the
        prediction :math:`\hat{x} = W_{out}^T r`, and `x` only sets the number of steps.

        Parameters
        ----------
        x : torch.Tensor of shape (nt, n_input)
            Input signal.
        nt_transient : int, default 500
            Number of steps to run with zero input before recording.
        closed_loop : bool, default True
            If True, feed the prediction back as input instead of using `x`.

        Returns
        -------
        s_rec : torch.Tensor of shape (nt, n_hidden)
            Network state at each step, recorded before the step's update.
        x_hat_rec : torch.Tensor of shape (nt, n_input)
            Prediction at each step. It stays all zeros in open loop.

        Raises
        ------
        ValueError
            If `closed_loop` is True and `n_input` differs from `n_output`.
        """
        nt = x.size(0)

        if closed_loop and self.n_input != self.n_output:
            raise ValueError("In closed loop model, `n_input` must equal `n_output`.")

        # Record the state of the system
        s_rec = torch.zeros((nt, self.n_hidden), **self.model.factory_kwargs)
        x_hat_rec = torch.zeros((nt, self.n_input), **self.model.factory_kwargs)
        # Transient Period
        print("Transient Period:")
        for i in tqdm(range(nt_transient)):
            self.model.forward(0)
        # Prepare the input signal current
        input_current = 0.0
        for i in tqdm(range(nt)):
            s_rec[i] = self.model.state().squeeze()  # Record the SNN state
            if closed_loop:
                x_hat = self.W_out.t() @ s_rec[i].reshape(self.n_hidden, 1)  # Feedback loop
                x_hat_rec[i] = x_hat.squeeze()
                input_current = self.W_in @ x_hat
            else:
                input_current = self.W_in @ x[i].reshape(self.n_input, 1)
            self.model.forward(input_=input_current)

        return s_rec, x_hat_rec

    def fit_echo_state(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        nt_transient: int = 500,
        ridge_reg: float = 1e-5,
    ) -> torch.Tensor:
        r"""Train `W_out` with ridge regression on the open-loop states.

        Runs :meth:`forward` in open loop on `x`, collects the states :math:`S`, and sets

        .. math::

            W_{out} = (S^T S + \lambda I)^{-1} S^T Y

        where :math:`Y` is `y` and :math:`\lambda` is `ridge_reg`.

        Parameters
        ----------
        x : torch.Tensor of shape (nt, n_input)
            Input signal.
        y : torch.Tensor of shape (nt, n_output)
            Target output.
        nt_transient : int, default 500
            Number of steps to run with zero input before recording.
        ridge_reg : float, default 1e-5
            Ridge regularization coefficient :math:`\lambda`.

        Returns
        -------
        torch.Tensor of shape (nt, n_output)
            Fitted output :math:`S W_{out}`.
        """
        s_rec, _ = self.forward(x, nt_transient=nt_transient, closed_loop=False)  # forward task for training is open loop
        Pinv = torch.linalg.pinv(s_rec.T @ s_rec + ridge_reg * torch.eye(self.n_hidden, **self.model.factory_kwargs))
        self.W_out.data = Pinv @ s_rec.T @ y
        y_hat = s_rec @ self.W_out

        return y_hat

    def fit_force(
        self, x: torch.Tensor, nt_transient: int = 500, ridge_reg: float = 1.0, rls_step: int = 20, ff_coeff: float = 1.0
    ) -> torch.Tensor:
        r"""Train `W_out` online with FORCE learning.

        The reservoir runs in closed loop and feeds its prediction :math:`\hat{x}` back as input. Every
        `rls_step` steps, a recursive least squares update moves `W_out` to reduce the error
        :math:`\hat{x} - x`. See :meth:`_rls` for the update.

        Parameters
        ----------
        x : torch.Tensor of shape (nt, n_output)
            Target signal for the prediction to follow.
        nt_transient : int, default 500
            Number of steps to run with zero input before training.
        ridge_reg : float, default 1.0
            Regularization :math:`\alpha` of the recursive least squares update. The first call starts `Pinv` at
            :math:`I / \alpha`. Later calls keep the existing `Pinv` and ignore this value.
        rls_step : int, default 20
            Number of steps between updates.
        ff_coeff : float, default 1.0
            Forgetting factor of the recursive least squares update, in the range (0, 1]. With 1.0, all past
            steps have equal weight.

        Returns
        -------
        torch.Tensor of shape (nt, n_output)
            Prediction at each step.

        Raises
        ------
        ValueError
            If `n_input` differs from `n_output`.
        """
        if self.n_input != self.n_output:
            raise ValueError("When using the FORCE learning method, `n_input` must equal `n_output`.")
        # Run transient period
        print("Transient Period")
        for _ in tqdm(range(nt_transient)):
            self.model.forward(input_=0)
        # Run main training loop
        nt = x.size(0)
        x_hat_rec = torch.zeros(size=(nt, self.n_output), **self.model.factory_kwargs)
        if self.Pinv is None:
            self.Pinv = torch.eye(self.model.N, **self.model.factory_kwargs) / ridge_reg

        x_hat = self.W_out.t() @ self.model.state()
        for i in tqdm(range(nt)):
            state = self.model.state()
            x_hat = self.W_out.t() @ state
            x_hat_rec[i] = x_hat.squeeze()
            input_current = self.W_in @ x_hat
            self.model.forward(input_=input_current)

            if i % rls_step == 0:
                self._rls(x[i].unsqueeze(1), x_hat, state, ff_coeff)
        return x_hat_rec

    def _rls(self, x: torch.Tensor, x_hat: torch.Tensor, state: torch.Tensor, ff_coeff: float = 1.0):
        r"""Apply one recursive least squares update to `W_out` and `Pinv`.

        With :math:`P` for `Pinv`, :math:`r` for `state`, and :math:`\lambda` for `ff_coeff`:

        .. math::

            u &= P r \\
            k &= \frac{u}{\lambda + r^T u} \\
            e &= \hat{x} - x \\
            P &\leftarrow \frac{P - k u^T}{\lambda} \\
            W_{out} &\leftarrow W_{out} - k e^T

        Parameters
        ----------
        x : torch.Tensor of shape (n_input, 1)
            Target at the current step.
        x_hat : torch.Tensor of shape (n_input, 1)
            Prediction at the current step.
        state : torch.Tensor of shape (n_hidden, 1)
            Network state that produced `x_hat`.
        ff_coeff : float, default 1.0
            Forgetting factor :math:`\lambda`. With 1.0, the memory is infinite.

        Returns
        -------
        torch.Tensor of shape (n_input, 1)
            The error :math:`e`.

        Raises
        ------
        ValueError
            If `x`, `x_hat`, or `state` has the wrong first dimension.
        TypeError
            If `Pinv` is None.
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
