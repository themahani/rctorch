[![codecov](https://codecov.io/gh/themahani/ml-force/graph/badge.svg?token=7UXVQC69IP)](https://codecov.io/gh/themahani/ml-force)
[![test](https://github.com/themahani/ml-force/actions/workflows/ci.yml/badge.svg)](https://github.com/themahani/ml-force)

# rcTorch

rcTorch is a Python package implementing reservoir computing architecures using pyTorch as its backbone.
This package includes various neuronal models built-in, but the reservoir can be created
using artificial neronal model built into pyTorch.

## Installation

### From PyPI

```bash
pip install rctorch
```

### From Git

```bash
git clone https://github.com/themahani/rcTorch.git
cd rcTorch
pip install -e .
```

## Usage

```python
from rctorch import Reservoir
from rctorch.models import LIF
from rctorch.supervisors import VanDerPol
from rctorch.utils import z_tranform

model_params = {
    "Ne": 100,  # Excitatory neurons
    "Ni": 100,  # Inhibitory neurons
    "dt": 1e-1, # Time step for integration
    "gbar": 25, # synaptic-coupling strength, can be engineered
    "BIAS": 75, # BIAS current for all neurons, can be engineered
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

T = 1000
x = VanDerPol(T=T, dt=model_params["dt"], mu=1.5, tau=0.02).generate(transient_time=100)
x = z_transform(x.T)    # shape (nt, 2) where `nt` is the number of time steps of generated data
x_tensor = torch.tensor(x, dtype=model_params["dtype"], device=model_params["device"])
dim = x_tensor.size(1)

res = Reservoir(
    model_cls = LIF,
    n_input = dim,
    n_ouput = dim,
    w_in_amp = 100,
    **model_params
)
t_transient = 200   # transient time in [ms]
nt_transient = int(t_transient / model_params["dt"])
x_hat = res.fit_force(
    x=x_tensor,                     # Input data (supervisor to train on)
    nt_transient=nt_transient,      # Transient time in time steps
    ridge_reg = 1.0,                # Ridge regularization coefficient
    rls_steps = 20,                 # gap between each RLS execution in unit of time steps (i.e every 20 time steps)
    ff_coef = 1 - 1e-4              # Forgetting factor for the RLS algorithm, must be in the range [0.0, 1.0]
)

```

## Documentation

Coming soon...

## License

This project is licensed under the MIT License - see the LICENSE file for details.
