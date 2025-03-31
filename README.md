# ML-Force

ML-Force is a Python package implementing Morris-Lecar neural networks with force learning.

## Installation

```bash
pip install ml-force
```

## Usage

```python
from ml_force import MorrisLecar, MorrisLecarCurrent

# Initialize network
ml = MorrisLecarCurrent(
    supervisor=signal,
    N=500,
    T=1000,
    dt=0.05
)

# Train network
...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
