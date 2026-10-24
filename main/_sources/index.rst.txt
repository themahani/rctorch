rcTorch
=======

rcTorch implements reservoir computing in PyTorch. The hidden layer of a :class:`~rctorch.Reservoir` is a
spiking neural network, such as a :class:`~rctorch.models.LIF` or :class:`~rctorch.models.MorrisLecar` network.
You can train the output weights with ridge regression or with FORCE learning, and run the trained reservoir in
closed loop to reproduce a target signal.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install rctorch

To work on the source, clone the repository and set up the environment with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/themahani/rctorch.git
   cd rctorch
   uv sync

.. toctree::
   :maxdepth: 2

   api
