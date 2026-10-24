#!/usr/bin/env python

"""Hyperparameter search for :class:`~rctorch.Reservoir`."""

import concurrent.futures
import inspect
import itertools
import json
import os
import threading
from typing import Any

import numpy as np
import torch

from .models import *
from .reservoir import Reservoir


class BruteForceMesh:
    """Grid search over reservoir parameters.

    Builds every combination of the values in `params`, trains one reservoir per combination with
    FORCE learning, tests it, and saves the results to disk. The simulations run in a thread pool.

    Parameters
    ----------
    reservoir_kwargs : dict
        Fixed keyword arguments for :class:`~rctorch.Reservoir`. They override entries of `params` with
        the same name.
    train_kwargs : dict
        Keyword arguments for :meth:`~rctorch.Reservoir.fit_force`.
    test_kwargs : dict
        Keyword arguments for :meth:`~rctorch.Reservoir.forward` in the test phase.
    params : dict of str to np.ndarray
        Parameter names mapped to 1D arrays of the values to try.
    num_threads : int, default 4
        Maximum number of simulations that run at the same time.
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
        """Return nested lists of None with the shape given by `dims`.

        Parameters
        ----------
        dims : list of int
            Length of each nesting level.

        Returns
        -------
        list
            Nested lists of None.
        """
        if len(dims) == 1:
            return [None] * dims[0]
        return [self._create_nested_list(dims[1:]) for _ in range(dims[0])]

    def run(self, save_dir: str) -> None:
        """Run one simulation per parameter combination and save each result.

        Each simulation writes to its own subdirectory of `save_dir`, named after its parameter values with
        dots replaced by ``p``. For example, ``w_in_amp=20, gbar=10`` gives ``w_in_amp_20p0000_gbar_10p0000``.
        The subdirectory holds ``reservoir_params.json``, ``output_data_train.npy``, and ``output_data_test.npy``.

        Parameters
        ----------
        save_dir : str
            Directory to save the results in.

        Notes
        -----
        This method waits for every simulation but does not re-raise their errors. A simulation that fails
        during training or testing prints its error. A simulation that fails while saving prints nothing.
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
        """Return the reservoir keyword arguments for one grid point.

        Parameters
        ----------
        indices : tuple of int
            Index into each parameter's array of values, in the order of `params`.

        Returns
        -------
        dict
            The selected parameter values merged with `reservoir_kwargs`.
        """
        reservoir_params = {self.param_names[i]: self.params[self.param_names[i]][index] for i, index in enumerate(indices)}
        reservoir_params = reservoir_params | self.reservoir_kwargs
        return reservoir_params

    def _run_and_save_simulation(self, reservoir_params: dict, save_dir: str) -> None:
        """Train and test one reservoir and save its outputs.

        Parameters
        ----------
        reservoir_params : dict
            Keyword arguments for :class:`~rctorch.Reservoir`.
        save_dir : str
            Base directory for the results.

        Raises
        ------
        Warning
            If the simulation or saving fails. The original exception is the cause.

        Notes
        -----
        This method only supports FORCE learning.
        """
        try:
            res = Reservoir(**reservoir_params)  # Instantiate the reservoir with all the specified params
            xhat_rec_train = res.fit_force(**self.train_kwargs)
            _s_rec_test, xhat_rec_test = res.forward(**self.test_kwargs)

        except Exception as e:
            print(f"Error running simulation with params {reservoir_params}: {e}")
            raise Warning(f"Error running simulation with params {reservoir_params}: {e}") from e

        try:
            # Construct a unique directory name based on parameters
            param_dir_name = "_".join(
                [f"{name}_{reservoir_params[name]:.4f}".replace(".", "p") for name in self.param_names]
            )  # replace . with p for directory names
            simulation_dir = os.path.join(save_dir, param_dir_name)
            os.makedirs(simulation_dir, exist_ok=True)
            # Save data for this simulation
            self._save_simulation_data(
                simulation_dir,
                reservoir_params,
                xhat_rec_train.cpu().numpy(),
                xhat_rec_test.cpu().numpy(),
            )

            print(f"Simulation with params {reservoir_params} saved to {simulation_dir}")

        except Exception as e:
            raise Warning(f"Coudln't save simulation results...\n{e}") from e

    def _save_simulation_data(
        self,
        save_path: str,
        reservoir_params: dict,
        output_data_train: np.ndarray,
        output_data_test: np.ndarray,
    ):
        """Save one simulation's parameters and outputs.

        Removes ``BIAS``, ``model_cls``, and ``device`` from `reservoir_params`, then writes the rest to JSON.

        Parameters
        ----------
        save_path : str
            Directory to write the files to.
        reservoir_params : dict
            Reservoir keyword arguments. This method modifies the dict.
        output_data_train : np.ndarray
            Output of :meth:`~rctorch.Reservoir.fit_force`.
        output_data_test : np.ndarray
            Prediction of :meth:`~rctorch.Reservoir.forward` in the test phase.
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
        """Store one simulation's parameters and output in the nested result lists.

        Nothing calls this method yet.

        Parameters
        ----------
        indices : tuple of int
            Position of the simulation in the grid.
        reservoir_params : dict
            Reservoir keyword arguments of the simulation.
        output_data : np.ndarray
            Output of the simulation.
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
    """JSON encoder for reservoir keyword arguments.

    Encodes numpy scalars as Python numbers. Encodes numpy arrays, torch devices and dtypes, classes,
    callables, and other objects as dicts with a ``"__type__"`` key that names the kind of value.
    """

    def default(self, o):
        # Handle numpy arrays
        if isinstance(o, np.ndarray):
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(o.dtype),
                "shape": o.shape,
                "data": o.tolist(),
            }

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
                "class": {
                    "module": o.__class__.__module__,
                    "name": o.__class__.__name__,
                },
                "attributes": o.__dict__,
            }

        # Fallback to default
        return super().default(o)


# TODO: Update PSO to match the new model architecture
class ParticleSwarmOptimizer:
    """Particle swarm optimizer for model parameters.

    Not implemented yet. :meth:`fitness` and :meth:`run` are empty.
    """

    def __init__(
        self,
        Model: MorrisLecar | object,
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
    def fitness(self, states: np.ndarray | torch.Tensor | np.number):
        pass

    # TODO: Implement the PSO algorithm
    def run(
        self,
        w_inertia: float = 0.7298,
        cognitive: float = 1.4944,
        social: float = 1.4944,
    ):
        pass
