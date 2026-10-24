#!/usr/bin/env python

"""A module that implements coordinate descent for the MorrisLecarBlock class"""

import concurrent.futures
import inspect
import itertools
import json
import os
import threading
from sys import argv
from typing import Any, Union

import numpy as np
import torch

from .models import *
from .reservoir import Reservoir


class BruteForceMesh:
    """
    BruteForceMesh is a brute-force algorithm developed with parallel computing
    functionality. It takes a set of hyper-parameters and their ranges to run
    through, and runs all simulations. It can run on multiple threads which can be
    configured using the `num_threads` parameter.

    Args:
        reservoir_kwargs (dict): Confgurations for the reservoir
        train_kwargs (dict): Arguments to be passed to the reservoir's `fit_force` method.
        test_kwargs (dict): Arguments to be passed to the reservoir's `forward` method for test phase prediction.
        params (dict[str, np.ndarray]): Dictionary of parameter names and their ranges as numpy arrays.
        num_threads (int, optional): The maximum number of threads to use for parallel simulations. Defaults to 4.
    """

    def __init__(
        self,
        reservoir_kwargs: dict,
        train_kwargs: dict,
        test_kwargs: dict,
        params: dict[str, np.ndarray],
        num_threads: int = 4,
    ) -> None:
        self.reservoir_kwargs = reservoir_kwargs
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
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
                future = executor.submit(self._run_and_save_simulation, reservoir_params, save_dir)
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

    def _run_and_save_simulation(self, reservoir_params: dict, save_dir: str) -> None:
        """
        Run a single simulation, save its data to a separate directory.

        Args:
            reservoir_params (dict): Parameters to initialize the reservoir with.
            indices (tuple): Indices corresponding to the parameter combination.
            save_dir (str): The base directory to save simulation data in.

        Notes:
            Currently only supports the FORCE learning method.

        """
        try:
            res = Reservoir(**reservoir_params)  # Instantiate the reservoir with all the specified params
            xhat_rec_train = res.fit_force(**self.train_kwargs)
            s_rec_test, xhat_rec_test = res.forward(**self.test_kwargs)

        except Exception as e:
            print(f"Error running simulation with params {reservoir_params}: {e}")
            raise Warning(f"Error running simulation with params {reservoir_params}: {e}")

        try:
            # Construct a unique directory name based on parameters
            param_dir_name = "_".join(
                [f"{name}_{reservoir_params[name]:.4f}".replace(".", "p") for name in self.param_names]
            )  # replace . with p for directory names
            simulation_dir = os.path.join(save_dir, param_dir_name)
            os.makedirs(simulation_dir, exist_ok=True)
            # Save data for this simulation
            self._save_simulation_data(
                simulation_dir, reservoir_params, xhat_rec_train.cpu().numpy(), xhat_rec_test.cpu().numpy()
            )

            print(f"Simulation with params {reservoir_params} saved to {simulation_dir}")

        except Exception as e:
            raise Warning(f"Coudln't save simulation results...\n{e}")

    def _save_simulation_data(
        self,
        save_path: str,
        reservoir_params: dict,
        output_data_train: np.ndarray,
        output_data_test: np.ndarray,
    ):
        """
        Save data for a single simulation to the specified path.

        Args:
            save_path (str): Directory to save the data in.
            reservoir_params (dict): Model parameters for this simulation.
            output_data_train (np.ndarray): Model output data.
            output_data_test (np.ndarray): Model output data.
        """
        params_fp = os.path.join(save_path, "reservoir_params.json")
        reservoir_params.pop("BIAS")
        reservoir_params.pop("model_cls")
        reservoir_params.pop("device")
        with open(params_fp, "w") as params_file:
            json.dump(reservoir_params, fp=params_file, cls=KWArgsEncoder)
        output_file = os.path.join(save_path, "output_data_train.npy")
        np.save(output_file, output_data_train)
        output_file = os.path.join(save_path, "output_data_test.npy")
        np.save(output_file, output_data_test)

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


class KWArgsEncoder(json.JSONEncoder):
    def default(self, o):
        # Handle numpy arrays
        if isinstance(o, np.ndarray):
            return {"__type__": "numpy.ndarray", "dtype": str(o.dtype), "shape": o.shape, "data": o.tolist()}

        # Handle numpy scalars
        if isinstance(o, (np.generic,)):
            return o.item()

        # Handle torch device type
        if isinstance(o, torch.device):
            return {"__type__": "torch.device", "device_type": o.type, "index": o.index}

        if isinstance(o, torch.dtype):
            return {"__type__": "torch.dtype", "dtype": str(o)}

        if isinstance(o, torch.device.type.__class__):
            # Rare case: direct DeviceType enum-like object
            return {"__type__": "torch.DeviceType", "value": str(o)}

        # Handle classes and types
        if inspect.isclass(o):
            return {"__type__": "class", "module": o.__module__, "name": o.__name__}

        # Handle functions or callables
        if callable(o):
            return {"__type__": "callable", "module": o.__module__, "name": o.__name__}

        # Handle objects from your custom classes
        if hasattr(o, "__class__") and not isinstance(
            o, (str, bytes, bytearray, dict, list, tuple, int, float, bool, type(None))
        ):
            return {
                "__type__": "object",
                "class": {"module": o.__class__.__module__, "name": o.__class__.__name__},
                "attributes": o.__dict__,
            }

        # Fallback to default
        return super().default(o)


# TODO: Update PSO to match the new model architecture
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
