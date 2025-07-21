#!/usr/bin/env python

"""A module that implements coordinate descent for the MorrisLecarBlock class"""

import concurrent.futures
import itertools
import json
import os
import threading
from json.encoder import JSONEncoder
from sys import argv
from typing import Any, Union

import numpy as np
import torch

from rctorch import reservoir

from .models import *
from .reservoir import Reservoir
from .utils import z_transform


class BruteForceMesh:
    """
    BruteForceMesh is brute-force algorithm developed with parallel computing
    functionality. It takes a set of hyper-parameters and their ranges to run
    through, and runs all simulations. It can run multiple threads which can be
    configured using the `num_threads` parameter.

    Args:
        reservoir_kwargs (dict): Confgurations for the reservoir
        render_args (dict): Arguments to be passed to the model's render method.
        params (dict[str, np.ndarray]): Dictionary of parameter names and their ranges as numpy arrays.
        num_threads (int, optional): The maximum number of threads to use for parallel simulations. Defaults to 4.
    """

    def __init__(
        self,
        reservoir_kwargs: dict,
        render_args: dict,
        params: dict[str, np.ndarray],
        num_threads: int = 4,
    ) -> None:
        self.reservoir_kwargs = reservoir_kwargs
        self.render_args = render_args
        self.params = params
        self.param_names = list(self.params.keys())
        self.num_threads = num_threads

        # Save the dimensions of the param ranges
        self.dimensions = []
        for name in self.param_names:
            self.dimensions.append(self.params[name].size)

        # Save the params for each simulation (initialized as None for now, filled during run)
        self.model_params_list = self._create_nested_list(self.dimensions)
        # Save model outputs (initialized as None)
        self.model_outputs = self._create_nested_list(self.dimensions)
        self.model_v_traces = self._create_nested_list(self.dimensions)
        self.model_decoders = self._create_nested_list(self.dimensions)

    def _create_nested_list(self, dims):
        """Create nested lists of size specified by `dims` using a recursive algorithm."""
        if len(dims) == 1:
            return [None] * dims[0]
        return [self._create_nested_list(dims[1:]) for _ in range(dims[0])]

    def run(self, save_dir: str) -> None:
        """
        Run the simulations for all parameter combinations in parallel and save data.

        Args:
            save_dir (str): The directory where simulation data will be saved.
        """
        dimensions = self.dimensions
        ranges = [range(dim) for dim in dimensions]
        all_indices = itertools.product(*ranges)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for indices in all_indices:
                reservoir_params = self._create_reservoir_params(indices)
                future = executor.submit(self._run_and_save_simulation, reservoir_params, indices, save_dir)
                futures.append(future)
                print(f"Submitted simulation with params: {reservoir_params}")

            # Wait for all simulations to complete
            concurrent.futures.wait(futures)
            print("All simulations completed.")

    def _create_reservoir_params(self, indices):
        """
        Create parameter dictionary for the model initialization based on indices.
        """
        reservoir_params = {self.param_names[i]: self.params[self.param_names[i]][index] for i, index in enumerate(indices)}
        reservoir_params = reservoir_params | self.reservoir_kwargs
        return reservoir_params

    def _run_and_save_simulation(self, reservoir_params: dict, indices, save_dir: str) -> None:
        """
        Run a single simulation, save its data to a separate directory.

        Args:
            model_params (dict): Parameters to initialize the model with.
            indices (tuple): Indices corresponding to the parameter combination.
            save_dir (str): The base directory to save simulation data in.

        Notes:
            Currently only supports the FORCE learning method.

        """
        try:
            res = Reservoir(**reservoir_params)  # Instantiate the reservoir with all the specified params
            xhat_rec = res.fit_force(**self.render_args)

            # Construct a unique directory name based on parameters
            param_dir_name = "_".join(
                [f"{name}_{reservoir_params[name]:.4f}".replace(".", "p") for name in self.param_names]
            )  # replace . with p for directory names
            simulation_dir = os.path.join(save_dir, param_dir_name)
            os.makedirs(simulation_dir, exist_ok=True)

            # Save data for this simulation
            self._save_simulation_data(simulation_dir, reservoir_params, xhat_rec.cpu().numpy())

            # Optionally, update the nested lists (if you need to access all results later in memory)
            # self._store_results_in_memory(indices, reservoir_params, xhat_rec)

            print(f"Simulation with params {reservoir_params} saved to {simulation_dir}")

        except Exception as e:
            print(f"Error running simulation with params {reservoir_params}: {e}")

    def _save_simulation_data(
        self,
        save_path: str,
        reservoir_params: dict,
        output_data: np.ndarray,
    ):
        """
        Save data for a single simulation to the specified path.

        Args:
            save_path (str): Directory to save the data in.
            model_params (dict): Model parameters for this simulation.
            output_data (np.ndarray): Model output data.
        """
        params_fp = os.path.join(save_path, "reservoir_params.json")
        with open(params_fp, "w") as params_file:
            json.dumps(reservoir_params, fp=params_file, cls=NumpyArrayEncoder)
        output_file = os.path.join(save_path, "model_outputs.npy")
        np.save(output_file, output_data)

    def _store_results_in_memory(self, indices, reservoir_params, output_data):
        """
        Store simulation results in the class's nested lists.
        """

        # Helper function to set value in nested list based on indices
        # TODO: Figure out if this implementation is wrong
        def set_nested_value(nested_list, indices, value):
            if not indices:
                return value
            current_index = indices[0]
            if len(indices) == 1:
                nested_list[current_index] = value
            else:
                set_nested_value(nested_list[current_index], indices[1:], value)

        set_nested_value(self.model_params_list, indices, reservoir_params)
        set_nested_value(self.model_outputs, indices, output_data)


class NumpyArrayEncoder(json.encoder.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.DeviceObjType):
            return str(o)
        return JSONEncoder.default(self, o)


class ParticleSwarmOptimizer:
    def __init__(
        self,
        Model: Union[MorrisLecar, object],
        render_args: dict[str, Any],
        default_args: dict[str, Any],
        n_particles: int = 10,
    ) -> None:
        self.n_particles = n_particles
        self.Model = Model
        self.render_args = render_args
        self.default_args = default_args
        self.param_names = [key for key, _ in self.model_params.items()]

    def _run_in_thread(self, model_params: dict[str, Any]) -> None:
        model = self.Model(**model_params)  # Instanciate the model with all the specified params
        _, v_trace, dec_trace = model.render(**self.render_args)  # Render the model with the specified render args

        # Save all relevant data of the model
        self.model_outputs = model.x_hat_rec.cpu().numpy()
        self.model_v_traces = v_trace.cpu().numpy()
        self.model_decoders = dec_trace.cpu().numpy()
        self.model_params_list = model_params

    def _simulate_parameter(self, state_params: dict[str, Any], particle_num: int) -> threading.Thread:
        # Create parameter dictionary for the model initialization
        model_params = state_params | self.default_args
        # Create and run the model in seperate thread
        thread = threading.Thread(
            target=self._run_in_thread,
            name=f"Thread {particle_num}",
            args=(model_params),
        )
        return thread

    # TODO: Implement the fitness function
    def fitness(self, states: Union[np.ndarray, torch.Tensor, np.number]):
        pass

    # TODO: Implement the PSO algorithm
    def run(
        self,
        w_inertia: float = 0.7298,
        cognitive: float = 1.4944,
        social: float = 1.4944,
    ):
        pass


class TestModel:
    def __init__(self, Q, gbar, dt) -> None:
        self.Q = Q
        self.dt = dt
        self.gbar = gbar
        self.n = 2000
        self.sup = torch.randn(self.n, 1)
        self.x_hat_rec = torch.randn(self.n, 1)

    def render(self, rls_stop):
        return 0, [0, 0, 0], rls_stop


def test():
    params = {"Q": np.arange(50), "gbar": np.arange(30)}
    render_args = {"rls_stop": 0}
    default_args = {"dt": 1.0}
    bfm = BruteForceMesh(TestModel, default_args, render_args, params)
    bfm.run()
    import os

    cwd = os.getcwd()
    resdir = os.path.join(cwd, "results", "test")
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    bfm.save_data(resdir, f_id="01")


def main():
    # Get the freq of the supervisor from the user
    # try:
    #     freq = float(input(">> Enter the frequency of the supvisor in [Hz]:\n>> "))
    # except:
    #     raise ValueError("You need to pass in a float as an argument.")

    seed = 1
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # Set the param ranges
    Q_range = np.linspace(0, 500, 20)
    gbar_range = np.linspace(0, 20, 10)
    # chop the mesh into portions to not reach time limit on ARC server nodes
    portion = int(argv[1])
    if portion >= 4:
        raise ValueError("Argument must be an integer from 0 to 3...")

    q_mid = int(Q_range.size // 2)
    gbar_mid = int(gbar_range.size // 2)
    q_idx = portion % 2
    gbar_idx = portion // 2
    if q_idx == 0:
        Q_range = Q_range[:q_mid]
    else:
        Q_range = Q_range[q_mid:]

    if gbar_idx == 0:
        gbar_range = gbar_range[:gbar_mid]
    else:
        gbar_range = gbar_range[gbar_mid:]

    bfm_params = {"Q": Q_range, "gbar": gbar_range}

    lamda = 1e-5

    from supervisors import LorenzAttractor

    # Global params for the model
    T = 12000
    dt = 1e-2

    x = LorenzAttractor(T, dt, tau=0.008).generate(transient_time=500.0)
    x = x.T
    signal = z_transform(x)

    NE = 200
    NI = 200
    N = NI + NE

    # input current for I and E neurons
    Ie = 75
    Ii = 75
    current = np.ones((N, 1))
    middle = N // 2
    current[:middle] *= Ie  # NE bias
    current[middle:] *= Ii  # NI bias

    # RLS params
    rls_start = round(T * 0.02)
    rls_start = 500
    rls_stop = round(T * 0.85)
    rls_step = 20

    default_args = {
        "T": T,
        "supervisor": signal,
        "BIAS": current,
        "dt": dt,
        "N": N,
        "l": lamda,
    }

    # device_choice = input(">> Enter device (GPU/CPU): ")
    device_choice = "GPU"
    if device_choice == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device <{device}> for PyTorch computations...\n")
        default_args = default_args | {"device": device}

    render_args = {
        "rls_start": rls_start,
        "rls_stop": rls_stop,
        "rls_step": rls_step,
        "live_plot": False,
        "plt_interval": 100,
        "n_neurons": 10,
        "save_all": False,
    }

    model = MorrisLecar
    bfm = BruteForceMesh(model, default_args, render_args, bfm_params)
    try:
        bfm.run(save_dir="./results")

    except Exception as e:
        raise RuntimeError("Run cancelled with error:\n", e)


if __name__ == "__main__":
    test()
