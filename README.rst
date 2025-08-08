==============================================================
radiosim |ci| |pre-commit| |codecov| |pypi| |zenodo| |license|
==============================================================

.. |ci| image:: https://github.com/radionets-project/radiosim/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/radionets-project/radiosim/actions/workflows/ci.yml?branch=main
   :alt: Test Status

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/radionets-project/radiosim/main.svg
   :target: https://results.pre-commit.ci/latest/github/radionets-project/radiosim/main
   :alt: pre-commit.ci status

.. |codecov| image:: https://codecov.io/github/radionets-project/radiosim/badge.svg
   :target: https://codecov.io/github/radionets-project/radiosim
   :alt: Code coverage

.. |pypi| image:: https://badge.fury.io/py/radiosim.svg
   :target: https://badge.fury.io/py/radiosim
   :alt: PyPI version

.. |zenodo| image:: https://zenodo.org/badge/337708657.svg
   :target: https://zenodo.org/badge/latestdoi/337708657
   :alt: Zenodo DOI

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/license/mit
   :alt: License: MIT



Simulation of radio skies to create astrophysical data sets.
This repository is part of the `Radionets Project <https://github.com/radionets-project>`__.


Installation
============

This repository is built as a python package. We recommend creating a mamba environment to handle the dependencies of all packages.
You can create one by running the following command in this repository:

.. code-block:: shell-session

   $ mamba env create --file=environment.yml

You can start the environment with

.. code-block:: shell-session

   $ mamba activate radiosim

after the installation.


Usage
=====

Radio Galaxy Simulations
------------------------

There are currently three supported simulation types:
1. ``survey`` full sky simulation
2. ``jet`` extended source
3. ``mojave`` MOJAVE like extended source

In the ``radiosim`` environment you can start the simulation with

.. code-block:: shell-session

   $ radiosim-simulate path/to/rc/file.toml

You can find an example file in ``rc/default_simulation.toml``.
The simulations will be saved as ``.h5`` files.


Protoplanetary Disk Simulations
-------------------------------

Simulating protoplanetary disks requires `pytorch <https://pytorch.org/>`__.
You can install radiosim with the required dependencies using `uv <https://docs.astral.sh/uv>`__:

.. code-block:: shell-session

   $ uv pip install "radiosim[torch]"


You can start the simulation with using the ``ppdisk`` CLI tool:

.. code-block:: shell-session

   $ radiosim-ppdisk path/to/rc/file.toml

You can find an example file in ``rc/default_ppdisks_simulation.toml``.
The simulations will be saved as ``.h5`` files.

The simulation parameters are described in the `ppdisks` module.
