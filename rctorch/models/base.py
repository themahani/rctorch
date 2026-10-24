"""Base class for spiking network models."""

from typing import ClassVar

import torch
from torch import nn


class SNNBase(nn.Module):
    r"""Base class for the spiking networks that form the hidden layer of a reservoir.

    The base model integrates :math:`\dot{v} = -v + I` with forward Euler steps. Subclasses replace
    :meth:`mem_dot` and :meth:`forward` with their own dynamics. :class:`~rctorch.Reservoir` also expects
    a subclass to set ``N``, the number of neurons, and to either set ``r`` or override :meth:`state`.

    Parameters
    ----------
    n_hidden : int
        Number of neurons.
    dt : float
        Integration time step.
    device : torch.device, default torch.device("cpu")
        Device for the model's tensors.
    dtype : torch.dtype, default torch.float32
        Data type for the model's tensors.

    Attributes
    ----------
    mem : torch.Tensor of shape (n_hidden, 1)
        Membrane potential of each neuron. It starts at zero.
    factory_kwargs : dict
        ``device`` and ``dtype`` keyword arguments for creating tensors that match the model.
    """

    __constants__: ClassVar[list[str]] = ["n_hidden"]
    n_hidden: int

    __variables__: ClassVar[list[str]] = ["mem", "rate"]
    mem: torch.Tensor
    rate: torch.Tensor

    def __init__(
        self,
        n_hidden: int,
        dt: float,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.factory_kwargs = nn.factory_kwargs({"device": device, "dtype": dtype})
        self.n_hidden = n_hidden
        self.dt = dt

        # Neuron Variables
        self.mem = torch.zeros(size=(n_hidden, 1), **self.factory_kwargs)

    def _init_mem(self) -> None:
        self.mem = torch.zeros(size=(self.n_hidden, 1))  # NOTE: Might have to add factory_kwargs

    def mem_dot(self, input_: torch.Tensor) -> torch.Tensor:
        r"""Return the time derivative of the membrane potential.

        Parameters
        ----------
        input_ : torch.Tensor of shape (n_hidden, 1) or float
            Input current.

        Returns
        -------
        torch.Tensor of shape (n_hidden, 1)
            :math:`\dot{v} = -v + I`.
        """
        return -self.mem + input_

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Advance the model by one time step.

        Parameters
        ----------
        input_ : torch.Tensor of shape (n_hidden, 1) or float
            Input current for this step.

        Returns
        -------
        torch.Tensor of shape (n_hidden, 1)
            The updated `mem`. This is the model's own tensor, not a copy.
        """
        if self.mem is None:
            self._init_mem()
        # Example update for `mem`
        self.mem += self.dt * self.mem_dot(input_)
        return self.mem

    def state(self) -> torch.Tensor:
        """Return a copy of the state that the reservoir reads out.

        Returns
        -------
        torch.Tensor of shape (n_hidden, 1)
            A copy of ``self.r``.
        """
        return self.r.clone()
