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

from .models import MorrisLecar
from .utils import z_transform


def rmse(x, x_hat):
    """Return the Root Mean Squared Error for `x` vs `x_hat`."""
    return np.sqrt(np.mean((x - x_hat) ** 2))


class CoordinateDescent:
    def __init__(
        self,
        model: object,
        default_args: dict,
        render_args: dict,
        params: dict,
        metric,
        limit: float,
    ) -> None:
        self.Model = model
        self.default_args = default_args
        self.render_args = render_args
        self.params = params
        self.param_names = list(self.params.keys())
        self.current_score = np.inf
        self.current_param = self.param_names[0]
        self.limit = limit
        self.metric = metric

        # Initalize the parameter history as a dict of lists
        self.param_history = dict()
        self.current_params = dict()
        for name in self.param_names:
            self.param_history[name] = []
            self.current_params[name] = np.random.choice(self.params[name])  # randomly choose the first param values

        self.score_history = list()
        self.param_scores = None

    def _simulate_param_range(self):
        param_range = self.params[self.current_param]
        n_values = len(param_range)
        self.param_scores = np.zeros(n_values)
        threads = []
        # Start the simulations in threads
        for i in range(n_values):
            param_value = param_range[i]
            threads.append(self._simulate_parameter(param_value, i))
            threads[-1].start()
            print(f"Thread started: {threads[-1].name}")

        # wait for all threads to finish to continue
        for thread in threads:
            thread.join()

        best_score, index = self._locate_best_score()
        return best_score, index

    def _locate_best_score(self):
        try:
            index = np.nanargmin(self.param_scores)
            best_score = self.param_scores[index]

        except ValueError:
            print(f"Current Param: {self.current_param}")
            print("All scores are NaN values. Moving to the next parameter")
            index = np.nan
            best_score = np.inf
        return best_score, index

    def _run_in_thread(self, model_params: dict, i: int) -> None:
        model = self.Model(**model_params)
        _, _, _ = model.render(**self.render_args)
        self.param_scores[i] = self.metric(model.sup, model.x_hat_rec)

    def _simulate_parameter(self, param_value: str, i: int) -> threading.Thread:
        # Create parameter dictionary for the model initialization
        model_params = self.current_params.copy()
        model_params[self.current_param] = param_value
        model_params = model_params | self.default_args
        thread = threading.Thread(target=self._run_in_thread, name=f"Thread {i}", args=(model_params, i))
        return thread

    def run(self) -> None:
        while True:
            param_index = self.param_names.index(self.current_param)  # find index of current param name
            self.current_param = self.param_names[param_index - 1]  # move to the previous param. Loops around the list

            print(f"Started range simulation for param {self.current_param}")
            best_score, index = self._simulate_param_range()  # Simulate param range in threads

            # Skip CD for this param bacause of all nan values
            if index == np.nan:
                continue

            # If the new score doesn't change much stop the CD optimizer
            if np.abs(best_score - self.current_score) < self.limit:
                break

            # Store the new score
            self.current_score = best_score
            self.score_history.append(self.current_score)

            # Store the new parameters
            best_param_value = self.params[self.current_param][index]
            self.current_params[self.current_param] = best_param_value
            for name in self.param_names:
                self.param_history[name].append(self.current_params[name])
            print(f"Last run score: {self.current_score}")

        return


class BruteForceMesh:
    def __init__(
        self,
        model: object,
        default_args: dict,
        render_args: dict,
        params: dict[str, np.ndarray],
        num_threads: int = 4,
    ) -> None:
        """
        Initialize the BruteForceMesh class.

        Args:
            model (object): The model class to simulate (e.g., MorrisLecarCurrent).
            default_args (dict): Default arguments to be passed to the model constructor.
            render_args (dict): Arguments to be passed to the model's render method.
            params (dict[str, np.ndarray]): Dictionary of parameter names and their ranges as numpy arrays.
            num_threads (int, optional): The maximum number of threads to use for parallel simulations. Defaults to 4.
        """
        self.Model = model
        self.default_args = default_args
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
                model_params = self._create_model_params(indices)
                future = executor.submit(self._run_and_save_simulation, model_params, indices, save_dir)
                futures.append(future)
                print(f"Submitted simulation with params: {model_params}")

            # Wait for all simulations to complete
            concurrent.futures.wait(futures)
            print("All simulations completed.")

    def _create_model_params(self, indices):
        """
        Create parameter dictionary for the model initialization based on indices.
        """
        model_params = {self.param_names[i]: self.params[self.param_names[i]][index] for i, index in enumerate(indices)}
        model_params = model_params | self.default_args
        return model_params

    def _run_and_save_simulation(self, model_params: dict, indices, save_dir: str) -> None:
        """
        Run a single simulation, save its data to a separate directory.

        Args:
            model_params (dict): Parameters to initialize the model with.
            indices (tuple): Indices corresponding to the parameter combination.
            save_dir (str): The base directory to save simulation data in.
        """
        try:
            model = self.Model(**model_params)  # Instantiate the model with all the specified params
            _, v_trace, dec_trace = model.render(**self.render_args)  # Render the model

            # Construct a unique directory name based on parameters
            param_dir_name = "_".join(
                [f"{name}_{model_params[name]:.4f}".replace(".", "p") for name in self.param_names]
            )  # replace . with p for directory names
            simulation_dir = os.path.join(save_dir, param_dir_name)
            os.makedirs(simulation_dir, exist_ok=True)

            # Save data for this simulation
            self._save_simulation_data(
                simulation_dir,
                model_params,
                model.x_hat_rec.cpu().numpy(),
                v_trace.cpu().numpy(),
                dec_trace.cpu().numpy(),
            )

            # Optionally, update the nested lists (if you need to access all results later in memory)
            self._store_results_in_memory(
                indices,
                model_params,
                model.x_hat_rec.cpu().numpy(),
                v_trace.cpu().numpy(),
                dec_trace.cpu().numpy(),
            )

            print(f"Simulation with params {model_params} saved to {simulation_dir}")

        except Exception as e:
            print(f"Error running simulation with params {model_params}: {e}")

    def _save_simulation_data(
        self,
        save_path: str,
        output_data,
        v_trace_data,
        decoder_data,
    ):
        """
        Save data for a single simulation to the specified path.

        Args:
            save_path (str): Directory to save the data in.
            model_params (dict): Model parameters for this simulation.
            output_data (np.ndarray): Model output data.
            v_trace_data (np.ndarray): Voltage trace data.
            decoder_data (np.ndarray): Decoder trace data.
        """
        output_file = os.path.join(save_path, "model_outputs.npy")
        np.save(output_file, output_data)

        vtrace_file = os.path.join(save_path, "model_v_traces.npy")
        np.save(vtrace_file, v_trace_data)

        dec_file = os.path.join(save_path, "model_decoders.npy")
        np.save(dec_file, decoder_data)

    def _store_results_in_memory(self, indices, model_params, output_data, v_trace_data, decoder_data):
        """
        Store simulation results in the class's nested lists.
        """

        # Helper function to set value in nested list based on indices
        def set_nested_value(nested_list, indices, value):
            if not indices:
                return value
            current_index = indices[0]
            if len(indices) == 1:
                nested_list[current_index] = value
            else:
                set_nested_value(nested_list[current_index], indices[1:], value)

        set_nested_value(self.model_params_list, indices, model_params)
        set_nested_value(self.model_outputs, indices, output_data)
        set_nested_value(self.model_v_traces, indices, v_trace_data)
        set_nested_value(self.model_decoders, indices, decoder_data)


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
