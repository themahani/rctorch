import torch
import torch.nn as nn


class SNNBase(nn.Module):

    __constants__ = ["n_hidden"]
    n_hidden: int

    __variables__ = ["mem", "rate"]
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
        r"""Calculate the first time derivative of the membrane potential.
        :param input_: Input current
        :type input_: torch.Tensor

        :returns:
            - mem_dot_ (torch.Tensor): The time derivative of :param:`self.mem`
        :rtype: torch.Tensor
        """
        return -self.mem + input_

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        r"""
        :param input_: Input tensor for the current time step.
        :type input_: torch.Tensor, shape (`nt`, )

        :returns:
            - output (torch.Tensor): Updated membrane potential
        :rtype: (torch.Tensor)
        """
        if self.mem is None:
            self._init_mem()
        # Example update for `mem`
        self.mem += self.dt * self.mem_dot(input_)
        return self.mem

    def state(self) -> torch.Tensor:
        r"""Return the state variable of the model.

        Returns
        -------
        torch.Tensor
            The state variable of the model.
        """
        return self.r.clone()
