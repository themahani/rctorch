[![codecov](https://codecov.io/gh/themahani/ml-force/graph/badge.svg?token=7UXVQC69IP)](https://codecov.io/gh/themahani/ml-force)
[![CI](https://github.com/themahani/rctorch/actions/workflows/ci.yml/badge.svg)](https://github.com/themahani/rctorch/actions/workflows/ci.yml)

# rcTorch

rcTorch is a PyTorch library for reservoir computing with spiking neural networks.
A `Reservoir` feeds an input signal into a network of spiking neurons through fixed random weights and reads the
output from the network state through a linear readout.
You train the readout with ridge regression (`fit_echo_state`) or online with FORCE learning (`fit_force`).

The package has these parts:

- `rctorch.models` has the spiking networks `LIF`, `MorrisLecar`, and `MorrisLecarCurrent`.
  To use another neuron model, subclass `SNNBase`.
- `rctorch.supervisors` generates target signals: `VanDerPol`, `LorenzAttractor`, and `HyperChaoticAttractor`.
- `rctorch.optimizers.BruteForceMesh` runs a grid search over reservoir parameters and saves each result to disk.

## Install rcTorch

rcTorch needs Python 3.11 or later.
It is not on PyPI, and the `rctorch` package on PyPI is a different project.
Install rcTorch from GitHub:

```bash
pip install git+https://github.com/themahani/rctorch.git
```

To add rcTorch to a uv project, run `uv add git+https://github.com/themahani/rctorch.git`.

## Train a reservoir on a Van der Pol oscillator

This example trains a reservoir of 100 Morris-Lecar neurons to follow a Van der Pol oscillator with FORCE learning.
It takes a few seconds on a CPU.

```python
import torch

from rctorch import Reservoir
from rctorch.models import MorrisLecarCurrent
from rctorch.supervisors import VanDerPol
from rctorch.utils import minmax_transform

torch.manual_seed(0)
dt = 0.1  # Integration time step in ms

# 2000 ms of the oscillator, rescaled to [0, 1] and then centered on zero
x = VanDerPol(T=2000, dt=dt, mu=1.5, tau=0.02).generate(transient_time=100)
x = torch.tensor(minmax_transform(x.T, zero_mean=True), dtype=torch.float32)  # Shape (20000, 2)

res = Reservoir(
    model_cls=MorrisLecarCurrent,
    n_input=2,
    n_output=2,
    w_in_amp=100,
    Ne=50,
    Ni=50,
    dt=dt,
    BIAS=75.0,
    gbar=10,
)
x_hat = res.fit_force(x, nt_transient=2000, rls_step=5)

# Mean squared error over the last 500 ms, relative to the variance of the signal
error = ((x_hat[-5000:] - x[-5000:]) ** 2).mean() / x.var(dim=0).mean()
print(f"Relative error: {error:.3f}")
```

`fit_force` first runs the network for 2000 steps with no input, then trains for one step per row of `x`.
It shows a progress bar for each part.
After training, the script prints a relative error of about 0.01.

To run on a GPU, pass `device=torch.device("cuda")` to `Reservoir` and create `x` on the same device.

For more advanced examples in research settings, please visit <https://github.com/themahani/ml_force_research>.

## Read the documentation

For the full documentation, please visit <https://themahani.github.io/rctorch/>.

## Develop rcTorch

Clone the repository and create the environment:

```bash
git clone https://github.com/themahani/rctorch.git
cd rctorch
uv sync
```

`uv sync` creates `.venv`, installs rcTorch in editable mode, and installs the `dev` dependency group.

Run the linter, the formatter, and the tests:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

Build the documentation for the working tree into `docs/_build/html`:

```bash
uv run --group docs sphinx-build docs docs/_build/html
```

To build every published version, run `uv run --group docs sphinx-multiversion docs docs/_build/html`.
It builds only committed branches and tags.

## License

rcTorch is under the MIT License. See [LICENSE](LICENSE).
