# radiosim
Simulation of radio skies to create astrophysical data sets.
This repository is part of the [`radionets-project`](https://github.com/radionets-project).

## Installation

This repository is built as a python package. We recommend creating a mamba environment to handle the dependencies of all packages.
You can create one by running the following command in this repository:
```
$ mamba env create -f environment.yml
```
You can start the environment with
```
$ mamba activate radiosim
```
after the installation.

## Usage

There are currently three supported simulation types: 
1. `survey` full sky simulation
2. `jet` extended source
3. `mojave` MOJAVE like extended source

In the `radiosim` environment you can start the simulation with
```
$ radiosim path/to/rc/file.toml
```
You can find an exemplary file in `rc/default_simulation.toml`.
The simulations will be saved as `.h5` files.