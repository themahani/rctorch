[![codecov](https://codecov.io/gh/themahani/ml-force/graph/badge.svg?token=7UXVQC69IP)](https://codecov.io/gh/themahani/ml-force)

# ML-Force

ML-Force is a Python package implementing Morris-Lecar neural networks with force learning.

## Installation

### Via pip
```bash
pip install ml-force
```

### Via Git
```bash
git clone https://github.com/themahani/ml-force.git
cd ml-force
pip install -e .    # Install in editable mode
```

## Usage

```python
from ml_force import MorrisLecar, MorrisLecarCurrent

# Initialize network
ml = MorrisLecarCurrent(
    supervisor=signal,
    Ne=500,
    Ni=500,
    T=1000,
    dt=0.05
)

# Train network and visualize the training process using the render method.
# Here we pass custom arguments to control aspects of the simulation and live plotting.

ml.render(
    rls_start=500,      # Start applying RLS after 500 ms
    rls_stop=700,       # Stop applying RLS at 700 ms
    rls_step=20,        # Update the RLS algorithm every 20 iterations
    live_plot=True,     # Enable live plotting during training
    plt_interval=300,   # Update the live plot every 300 ms
    n_neurons=10,       # Record voltage traces for 10 selected neurons
    save_all=False      # Record traces only for the selected neurons (False)
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
