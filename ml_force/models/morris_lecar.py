#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from typing import Union, Any


class MorrisLecar:
    def __init__(
        self,
        supervisor: np.ndarray,
        dt: float,
        T: float,
        BIAS: Union[np.ndarray, torch.Tensor, np.float64],
        N: int = 20,
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
        Q: float = 100,
        l: float = 1e-5,
        gbar: float = 1,
        w_rand: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """A modified version of the Morris Lecar neural network model. This model uses a block structure to
        simplify and optimize for matrix multiplications using `Numpy`.

        Parameters
        ----------
        supervisor : np.ndarray
            The target output of the model. This is the signal the model is supposed to follow after being trained
        dt : float
            Time step for the neural networks and the supervisor in [ms].
        T : float
            The total simulation time of the system.
        BIAS : np.ndarray
            The input bias into all the neurons in the reservoir in [pA].
            Can be engineered.
        N : int, optional
            The number of neurons in the reservoir, by default 20
        C : float, optional
            The capaticance of the Morris Lecar model. Also acts as the time constant of the nerwork, by default 20
        g_L : float, optional
            Leak conductance, by default 2
        g_K : float, optional
            Potassium conductance, by default 8
        g_Ca : float, optional
            Calsium conductance, by default 4.4
        E_L : float, optional
            Baseline voltage for Leak, by default -60
        E_K : float, optional
            Baseline voltage for K, by default -84
        E_Ca : float, optional
            Baseline voltage for Calsium, by default 120
        v1 : float, optional
            Voltage param for the model dynamics, by default -1.2
        v2 : float, optional
            Voltage param for the model dynamics, by default 18
        v3 : float, optional
            Voltage param for the model dynamics, by default 2
        v4 : float, optional
            Voltage param for the model dynamics, by default 30
        phi : float, optional
            Frequency in `n_dot`, by default 0.04
        a_r : float, optional
            Synaptic rise time constant, by default .2
        a_d : float, optional
            Synaptic decay time constant, by default 0.02
        v_t : float, optional
            Constant for `T`, by default 2
        k_p : float, optional
            Constant for `T`, by default 5
        t_max : float, optional
            Constant for `T`, by default 1.0
        E_AMPA : float, optional
            The excitatory resting potential in [mV], by default 0
        E_GABA : float, optional
            The inhibitory resting potential in [mV], by default -75
        Q : float, optional
            The encoding strength coefficient, by default 2
        l : float, optional
            The learning strength for FORCE training, by default 2.0
        gbar : float, optional
            Synaptic conductance for all neurons in [nS], by default 1
        """
        # Morris Lecar model parameters
        self.device = device
        self._N = N
        self._BIAS = torch.tensor(BIAS, device=self.device)
        self._C = C
        self._g_L = g_L
        self._g_K = g_K
        self._g_Ca = g_Ca
        self._E_L = E_L
        self._E_K = E_K
        self._E_Ca = E_Ca
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
        self._E_AMPA = E_AMPA
        self._E_GABA = E_GABA
        self.gbar = gbar

        # Network connections
        self.v = torch.zeros(size=(N, 1), dtype=torch.float32, device=self.device)
        self.s = torch.zeros(size=(N, 1), dtype=torch.float32, device=self.device)
        self.n = torch.rand(N, 1, dtype=torch.float32, device=self.device)
        self.w = torch.rand(N, N, device=self.device) < (2 / self._N)
        self.E = torch.ones(size=(N, 1), dtype=torch.float32, device=self.device)
        middle = int(N // 2)
        self.E[:middle, 0] *= E_AMPA  # first half of the neurons are excitatory
        self.E[middle:, 0] *= E_GABA  # second half is inhibitory
        # Let's follow Dale's law
        self.ones = torch.ones(size=(1, self._N), dtype=torch.float32, device=self.device)
        self.E = (self.E @ self.ones).T
        # self.E = cp.tile(self.E, reps=(N, 1))
        self.w_rand = w_rand

        # Time series
        self.sup = torch.tensor(supervisor, dtype=torch.float32, device=self.device)
        self._nt = self.sup.shape[0]
        self._dt = dt
        self._duration = T
        self.time = torch.arange(0, self._nt, device=self.device) * self._dt
        self.time_cpu = self.time.cpu()
        self.exp = None
        # Encoding and decoding
        self.l = l
        dim = np.min(self.sup.shape)
        self.dec = torch.zeros(size=(N, dim), dtype=torch.float32, device=self.device)
        self.eta = Q * (2 * torch.rand(N, dim, dtype=torch.float32, device=self.device) - 1)
        self.Pinv = torch.eye(self._N, dtype=torch.float32, device=self.device) / self.l
        self.x_hat = self.dec.T @ self.s
        self.x_hat_rec = torch.zeros(
            size=(self._nt, self.x_hat.size()[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self.ipsc = torch.zeros(size=(N, 1), dtype=torch.float32, device=self.device)

    def __reinit__(self):
        self.Pinv = torch.eye(self._N, dtype=torch.float32, device=self.device) / self.l
        self.x_hat_rec = torch.zeros(
            size=(self._nt, self.x_hat.size()[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self.v = torch.zeros(size=(self._N, 1), dtype=torch.float32, device=self.device)
        self.s = torch.zeros(size=(self._N, 1), dtype=torch.float32, device=self.device)
        self.n = torch.rand(self._N, 1, device=self.device)

    def m_ss(self) -> torch.Tensor:
        return 0.5 * (1 + torch.tanh((self.v - self._v1) / self._v2))

    def n_ss(self) -> torch.Tensor:
        return 0.5 * (1 + torch.tanh((self.v - self._v3) / self._v4))

    def tau_n(self) -> torch.Tensor:
        cosh = torch.cosh((self.v - self._v3) / (2 * self._v4))
        # if cosh.isnan().any():
        #     raise ValueError("Found NaN value in tau_n")

        return 1 / (self._phi * cosh)

    def T(self):
        self.exp = torch.exp(-(self.v - self._v_t) / self._k_p)
        # if self.exp.isnan().any():
        #     raise ValueError("Found NaN value in exp")

        return self._t_max / (1 + self.exp)

    def s_dot(self):
        return self._a_r * self.T() * (1 - self.s) - self._a_d * self.s

    def n_dot(self):
        return (self.n_ss() - self.n) / self.tau_n()

    def calc_ipsc(self) -> None:
        self.ipsc = -self.gbar * self.w * ((self.v @ self.ones) - self.E) @ self.s

    def check_nan(self, inp: torch.Tensor, name: str):
        if inp.isnan().any():
            raise ValueError("Found NaN value in " + name)

    def v_dot(self, closed_loop: bool = True):
        self.calc_ipsc()  # Calculate the new post-synaptic potential
        I_L = -self._g_L * (self.v - self._E_L)
        # self.check_nan(I_L, "Leak current")
        I_K = -self._g_K * self.n * (self.v - self._E_K)
        # self.check_nan(self.n, "n")
        # self.check_nan(I_K, "Potassium current")
        I_Ca = -self._g_Ca * self.m_ss() * (self.v - self._E_Ca)
        # self.check_nan(I_Ca, "Calcium current")
        encoding = 0
        if closed_loop:
            encoding = self.eta @ self.x_hat

        # self.check_nan(encoding, "encoding")
        # print(encoding.shape)
        return (self._BIAS + I_L + I_K + I_Ca + self.ipsc + encoding) / self._C

    def _mask_inf(self, inp: torch.Tensor, inf_value: float = 1e5, mask_value=1e5):
        is_plus_inf = inp > inf_value
        is_minus_inf = inp < -inf_value

        if not is_minus_inf.any() and not is_plus_inf.any():  # If don't have inf values, don't do anything
            return

        inp[is_plus_inf] = mask_value
        inp[is_minus_inf] = -mask_value

        return inp

    def euler_step(self, closed_loop: bool = True, voltage_bound: float = None) -> None:
        dv = self._dt * self.v_dot(closed_loop)  # + self.w_rand * torch.rand(self._N, 1, device=self.device)
        self.n += self._dt * self.n_dot()
        self.s += self._dt * self.s_dot()
        self.v += dv

        if voltage_bound is not None:
            self._mask_inf(self.v, voltage_bound, voltage_bound)
        # self._mask_inf(self.n, 1, 1)

        self.x_hat = self.dec.T @ self.s

        if self.v.isnan().any():
            raise RuntimeError("NaN value encountered during iteration.")

    def _standardize(self, signal: torch.Tensor, i: int):
        """Scale the signal for have a range of [0, 1]

        Parameters
        ----------
        signal : torch.Tensor
            Input signal to transform
        i : int
            The index added to the signal

        Returns
        -------
        torch.Tensor
            The scaled signal
        """
        minim = torch.min(signal)
        maxim = torch.max(signal)
        signal = (signal - minim) / (maxim - minim) + i
        return signal

    def _update_rls_plot(self, i):
        self.rls_line.set_xdata(self.time_cpu[:i])
        self.rls_line.set_ydata(self.x_hat_rec[:i, -1].cpu())
        self.rls_sup_line.set_xdata(self.time_cpu[:i])
        self.rls_sup_line.set_ydata(self.sup[:i, -1].cpu())
        self.rls_ax.set_xlim(0, self.time_cpu[i])
        # self.ax.relim()
        # self.ax.autoscale(True, True, True)
        self.rls_fig.canvas.flush_events()
        self.rls_fig.canvas.restore_region(self.rls_bg)
        self.rls_ax.draw_artist(self.rls_line)
        self.rls_ax.draw_artist(self.rls_sup_line)
        self.rls_fig.canvas.blit(self.rls_ax.bbox)

    def _update_neural_plot(self, voltage_trace, time_index: int):
        for line_index in range(len(self.neural_lines)):
            signal = self._standardize(voltage_trace[:, line_index], line_index)
            self.neural_lines[line_index].set_xdata(self.time_cpu[:time_index])
            self.neural_lines[line_index].set_ydata(signal)
        self.neural_ax.set_xlim(0, self.time_cpu[time_index])
        # self.ax.relim()
        # self.ax.autoscale(True, True, True)
        self.neural_fig.canvas.flush_events()
        self.neural_fig.canvas.restore_region(self.neural_bg)

        for line in self.neural_lines:
            self.neural_ax.draw_artist(line)

        self.neural_fig.canvas.blit(self.neural_ax.bbox)

    def _update_decoder_plot(self, decoder_trace, time_index):
        for line_index in range(len(self.decoder_lines)):
            # signal = self._standardize(decoder_trace[:, line_index], line_index)
            self.decoder_lines[line_index].set_xdata(self.time_cpu[:time_index])
            self.decoder_lines[line_index].set_ydata(decoder_trace[:, line_index])
        self.decoder_ax.set_xlim(0, self.time_cpu[time_index])
        # self.ax.relim()
        # self.ax.autoscale(True, True, True)
        self.decoder_fig.canvas.flush_events()
        self.decoder_fig.canvas.restore_region(self.decoder_bg)

        for line in self.decoder_lines:
            self.decoder_ax.draw_artist(line)

        self.decoder_fig.canvas.blit(self.decoder_ax.bbox)

    def _rls_plot_init(self, rls_step):
        self.rls_fig = plt.figure("RLS", figsize=(10, 6))
        self.rls_ax = self.rls_fig.add_subplot(1, 1, 1)
        (self.rls_line,) = self.rls_ax.plot([], [], label="decoder")
        (self.rls_sup_line,) = self.rls_ax.plot([], [], label="supervisor")
        self.rls_ax.set_title(f"N={self._N}, T={self._duration}, RLS step = {rls_step}")
        self.rls_ax.set_ylim(-2, 2)
        self.rls_ax.set_xlim(0, 1)
        self.rls_bg = self.rls_fig.canvas.copy_from_bbox(self.rls_ax.bbox)

    def _neural_plot_init(self, voltage_trace, rls_step):
        self.neural_fig = plt.figure("Neural Signal", figsize=(10, 6))
        self.neural_ax = self.neural_fig.add_subplot(1, 1, 1)
        self.neural_lines = []
        n_neurons = voltage_trace.shape[1]
        for i in range(n_neurons):
            signal = voltage_trace[:, i]
            signal = self._standardize(signal, i)
            (line,) = self.neural_ax.plot([], [])
            self.neural_lines.append(line)

        self.neural_ax.set_title(f"N={self._N}, T={self._duration}, RLS step = {rls_step}")
        self.neural_ax.set_ylim(-1, n_neurons)
        self.neural_ax.set_xlim(0, 1)
        self.neural_bg = self.neural_fig.canvas.copy_from_bbox(self.neural_ax.bbox)

    def _decoder_plot_init(self, decoder_trace, rls_step):
        self.decoder_fig = plt.figure("Decoder Signal", figsize=(10, 6))
        self.decoder_ax = self.decoder_fig.add_subplot(1, 1, 1)
        self.decoder_lines = []
        n_neurons = decoder_trace.shape[1]

        for i in range(n_neurons):
            signal = decoder_trace[:, i]
            (line,) = self.decoder_ax.plot([], [])
            self.decoder_lines.append(line)

        self.decoder_ax.set_title(f"N={self._N}, T={self._duration}, RLS step = {rls_step}")
        self.decoder_ax.set_ylim(-50, 50)
        self.decoder_ax.set_xlim(0, 1)
        self.decoder_bg = self.decoder_fig.canvas.copy_from_bbox(self.decoder_ax.bbox)

    def _save_live_plots(self, save_dir: str = "plots"):
        """Save the live plots to the given directory and close the figures.
        If the directory does not exist, it will be created.
        The figures are saved as jpg files with the names "block_rls_output.jpg", "block_neural_output.jpg",
        and "block_decoder_output.jpg".

        Parameters
        ----------
        save_dir : str, optional
            directory to save the live plots, by default "plots"
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.rls_fig.savefig(os.path.join(save_dir, "block_rls_output.jpg"), bbox_inches="tight", dpi=250)
        self.neural_fig.savefig(os.path.join(save_dir, "block_neural_output.jpg"), bbox_inches="tight", dpi=250)
        self.decoder_fig.savefig(os.path.join(save_dir, "block_decoder_output.jpg"), bbox_inches="tight", dpi=250)
        plt.close(self.rls_fig)
        plt.close(self.neural_fig)
        plt.close(self.decoder_fig)

    def render(
        self,
        rls_start,
        rls_stop,
        rls_step,
        live_plot: bool,
        plt_interval: float,
        n_neurons: int = 10,
        save_all: bool = False,
    ):
        """
        Run the system simulation with the force method.

        This function simulates the model dynamics using an Euler integration scheme while recording the
        voltage of selected neurons and the corresponding decoder outputs over time. It optionally updates
        live plots during the simulation and saves these plots to a designated directory upon completion.

        rls_start : float or int
            The time (in ms) to start applying the Recursive Least Squares (RLS) algorithm. This value is
            converted from milliseconds to simulation time steps using the model's internal time step (self._dt).
        rls_stop : float or int
            The time (in ms) to stop applying the RLS algorithm. This value is also converted from milliseconds
            to simulation time steps.
        rls_step : int
            The interval (in simulation steps) at which the RLS algorithm is applied between rls_start and rls_stop.
            Determines whether live plotting is enabled. If True, the function will update plots during simulation.
            The interval (in ms) at which the live plots are updated; this value is converted to simulation time steps.
            The number of neurons to sample for recording voltage and decoder traces when save_all is False (default is 10).
            If True, voltage and decoder traces for all neurons are recorded; otherwise, only the sampled neurons are recorded (default is False).

        tuple of torch.Tensor
            A tuple containing:
              - random_neuron: A tensor containing the indices of the randomly selected neurons.
              - voltage_trace: A tensor containing the recorded voltage traces. Its shape is
                (number of time steps, n_neurons) if save_all is False, or (number of time steps, total neurons)
                if save_all is True.
              - decoder_trace: A tensor containing the recorded decoder outputs with a shape similar to voltage_trace.

        Raises
        ------
        Exception
            If an error is encountered during the Euler integration step, the simulation loop is terminated and the
            exception is printed.
        Exception
            If an error occurs while attempting to save the live plots after the simulation.
        """
        random_neuron = torch.tensor(
            np.random.choice(a=self._N, size=n_neurons, replace=False),
            device=self.device,
        )
        if save_all == False:
            voltage_trace = torch.zeros(
                size=(self.time.size()[0], n_neurons),
                dtype=torch.float32,
                device=self.device,
            )
            decoder_trace = torch.zeros(
                size=(self.time.size()[0], n_neurons),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            voltage_trace = torch.zeros(
                size=(self.time.size()[0], self._N),
                dtype=torch.float32,
                device=self.device,
            )
            decoder_trace = torch.zeros(
                size=(self.time.size()[0], self._N),
                dtype=torch.float32,
                device=self.device,
            )
        # Setup for RLS -- convert ms to number of time steps
        rls_start = int(rls_start // self._dt)
        rls_stop = int(rls_stop // self._dt)

        # Setup Plots
        if live_plot:
            plt_interval = int(plt_interval // self._dt)
            self._rls_plot_init(rls_step)
            self._neural_plot_init(voltage_trace.cpu(), rls_step)
            self._decoder_plot_init(decoder_trace.cpu(), rls_step)
            print(decoder_trace.shape)
            plt.show(block=False)

        # Main loop
        for i in tqdm(range(self._nt)):
            try:
                self.euler_step()

            except Exception as e:
                print(f"Error encountered. Looped stopped at iteration i={i}")
                print(e)
                break

            if save_all == False:
                voltage_trace[i] = self.v[random_neuron, 0]
                decoder_trace[i] = self.dec[random_neuron, -1]
            else:
                voltage_trace[i] = self.v[:, 0]
                decoder_trace[i] = self.dec[:, -1]

            self.x_hat_rec[i] = self.x_hat[:, 0]

            if live_plot and (i % plt_interval == 1):
                self._update_rls_plot(i)
                self._update_neural_plot(voltage_trace[:i].cpu(), i)
                self._update_decoder_plot(decoder_trace[:i].cpu(), i)
            if i > rls_start and i < rls_stop:
                if i % rls_step == 1:
                    self.rls(i)
        if live_plot:
            try:
                self._save_live_plots(save_dir=os.path.join(os.getcwd(), "plots"))
            except Exception as e:
                print("Failed to Save plots.\n", e)

        return random_neuron.cpu(), voltage_trace.cpu(), decoder_trace.cpu()

    def rls(self, i):
        """Run the system with the force method and update the decoder weights."""
        error = self.x_hat - self.sup[i].reshape(-1, 1)
        q = self.Pinv @ self.s
        self.Pinv -= (q @ q.T) / (1 + self.s.T @ q)
        self.dec -= (self.Pinv @ self.s) @ error.T

    def _train(self, rls_stop: float, rls_step: int, transient_time: float = 200):
        rls_stop = int(rls_stop // self._dt)  # Transform from [ms] to # of time steps
        nt = int(transient_time // self._dt)  # Transform from [ms] to # of time steps
        # Run the reservoir for a transient time
        print("Transient Time:")
        for i in tqdm(range(nt)):
            self.euler_step(closed_loop=False)

        # Run the main training loop
        print("Training time:")
        for i in tqdm(range(rls_stop)):
            self.euler_step(closed_loop=True)
            self.x_hat_rec[i] = self.x_hat[:, 0]

            if i % rls_step == 1:
                self.rls(i)

    def _infer(self, rls_stop: float) -> None:
        test_start = int(rls_stop // self._dt)  # Transform from [ms] to # of time steps

        for i in tqdm(range(test_start, self._nt)):
            self.euler_step(closed_loop=True)
            self.x_hat_rec[i] = self.x_hat[:, 0]

    @classmethod
    def _standardize(self, signal: torch.Tensor, i: int):
        """Scale the signal for have a range of [0, 1] + i for plotting.

        Parameters
        ----------
        signal : torch.Tensor
            Input signal to transform
        i : int
            The index added to the signal

        Returns
        -------
        torch.Tensor
            The scaled signal
        """
        minim = torch.min(signal)
        maxim = torch.max(signal)
        signal = (signal - minim) / (maxim - minim) + i
        return signal


class MorrisLecarCurrent(MorrisLecar):
    def __init__(
        self,
        supervisor: np.ndarray,
        dt: float,
        T: float,
        BIAS: Union[np.ndarray, np.float64],
        N: int = 20,
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
        Q: float = 100,
        l: float = 1e-5,
        gbar: float = 1,
        w_rand: float = 0.0,
        device: torch.device = torch.device("cpu"),
        p_sparsity: float = 1.0,
    ) -> None:
        """A modified version of the Morris Lecar neural network model. This model uses a block structure to
        simplify and optimize for matrix multiplications using `PyTorch`.

        Parameters
        ----------
        supervisor : np.ndarray
            The target output of the model. This is the signal the model is supposed to follow after being trained
        dt : float
            Time step for the neural networks and the supervisor in [ms].
        T : float
            The total simulation time of the system.
        BIAS : np.ndarray
            The input bias into all the neurons in the reservoir in [pA].
            Can be engineered.
        N : int, optional
            The number of neurons in the reservoir, by default 20
        C : float, optional
            The capaticance of the Morris Lecar model. Also acts as the time constant of the nerwork, by default 20
        g_L : float, optional
            Leak conductance, by default 2
        g_K : float, optional
            Potassium conductance, by default 8
        g_Ca : float, optional
            Calsium conductance, by default 4.4
        E_L : float, optional
            Baseline voltage for Leak, by default -60
        E_K : float, optional
            Baseline voltage for K, by default -84
        E_Ca : float, optional
            Baseline voltage for Calsium, by default 120
        v1 : float, optional
            Voltage param for the model dynamics, by default -1.2
        v2 : float, optional
            Voltage param for the model dynamics, by default 18
        v3 : float, optional
            Voltage param for the model dynamics, by default 2
        v4 : float, optional
            Voltage param for the model dynamics, by default 30
        phi : float, optional
            Frequency in `n_dot`, by default 0.04
        a_r : float, optional
            Synaptic rise time constant, by default .2
        a_d : float, optional
            Synaptic decay time constant, by default 0.02
        v_t : float, optional
            Constant for `T`, by default 2
        k_p : float, optional
            Constant for `T`, by default 5
        t_max : float, optional
            Constant for `T`, by default 1.0
        E_AMPA : float, optional
            The excitatory resting potential in [mV], by default 0
        E_GABA : float, optional
            The inhibitory resting potential in [mV], by default -75
        Q : float, optional
            The encoding strength coefficient, by default 2
        l : float, optional
            The learning strength for FORCE training, by default 2.0
        gbar : float, optional
            Synaptic conductance for all neurons in [nS], by default 1
        """
        # Morris Lecar model parameters
        super().__init__(
            supervisor,
            dt,
            T,
            BIAS,
            N,
            C,
            g_L,
            g_K,
            g_Ca,
            E_L,
            E_K,
            E_Ca,
            v1,
            v2,
            v3,
            v4,
            phi,
            a_r,
            a_d,
            v_t,
            k_p,
            t_max,
            E_AMPA,
            E_GABA,
            Q,
            l,
            gbar,
            w_rand,
            device,
        )

        # Network connections -- Normal Distribution
        self.w = torch.normal(
            mean=0,
            std=1 / np.sqrt(self._N * p_sparsity),
            size=(self._N, self._N),
            device=self.device,
            dtype=torch.float32,
        )

    def calc_ipsc(self) -> None:
        self.ipsc = -self.gbar * self.w @ self.s


def z_transform(signal: np.ndarray):
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)


def minmax_transform(signal, zero_mean: bool = False):
    min = signal.min(axis=0)
    max = signal.max(axis=0)
    minmaxed = (signal - min) / (max - min)
    if zero_mean:
        minmaxed -= minmaxed.mean(axis=0)
    return minmaxed


def test():
    """Test the module"""
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from plots import plot_model
    from supervisors import LorenzAttractor, VanDerPol

    seed = 1
    np.random.seed(seed)
    T = 1000
    dt = 5e-2
    t = np.arange(0, T, dt)
    nt = t.size
    x = LorenzAttractor(T, dt, tau=0.01).generate(transient_time=2000.0)

    x = x.T
    signal = z_transform(x)
    print(signal.shape)

    NE = 100
    NI = 100
    N = NI + NE

    # input current for I and E neurons
    Ie = 75
    Ii = 75
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias
    # current = np.random.rand(N, 1) * Ie

    # RLS params
    rls_start = round(T * 0.02)
    rls_start = 500  # ms
    rls_stop = round(T * 0.7)
    rls_step = 20
    Q = 200
    lamda = 0.8
    gbar = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device <{device}> for PyTorch computations...\n")
    torch.cuda.random.manual_seed(seed)

    try:
        model = MorrisLecar(
            supervisor=signal,
            BIAS=current,
            T=T,
            dt=dt,
            N=N,
            Q=Q,
            gbar=gbar,
            l=lamda,
            device=device,
        )
    except Exception as e:
        print(e)
        print(torch.cuda.memory_summary(device=device))

    start = time.time()
    random_neurons, voltage_trace, decoder_trace = model.render(
        rls_start=rls_start,
        rls_stop=rls_stop,
        rls_step=rls_step,
        live_plot=True,
        plt_interval=300,
        n_neurons=10,
        save_all=False,
    )
    end = time.time()
    print(f"Render took {end - start} seconds to finish...\n")

    plot_params = {
        "rls_start": rls_start,
        "rls_stop": rls_stop,
        "rls_step": rls_step,
        "lamda": lamda,
        "Q": Q,
        "n_vars": min(model.sup.shape),
    }
    save_dir = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_params["save_dir"] = save_dir
    plot_model(model, "lorenz", random_neurons, voltage_trace, decoder_trace, **plot_params)


if __name__ == "__main__":
    test()
