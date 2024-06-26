# README

This repository contains minimal code for running the numerical examples in the paper:

D. Pradovera, M. Nonino, and I. Perugia, _Geometry-based approximation of waves in complex domains_ (2023)

Preprint publicly available [here](https://arxiv.org/abs/2301.13613)!

## Prerequisites
* **numpy** and **scipy**
* **matplotlib**
* **fenics** and **mshr** for 2D FEM tests

## Fenics
The 2D FEM engine relies on [FEniCS](http://fenicsproject.org/). If you do not have FEniCS installed, you may want to create an [Anaconda3/Miniconda3](http://anaconda.org/) environment using the command
```
conda create -n fenicsenv -c conda-forge numpy=1.21.4=py38he2449b9_0 scipy=1.5.3=py38h828c644_0 matplotlib=3.4.3=py38h578d9bd_1 fenics=2019.1.0=py38_9 mshr=2019.1.0=py38hf9f41d3_3 boost-cpp=1.72.0=h312852a_5
```
This will create an environment where FEniCS (and all other required modules) can be used. In order to use FEniCS, the environment must be activated through
```
conda activate fenicsenv
```
See the [Anaconda documentation](http://docs.conda.io/) for more information.

### Fenics and mshr versions
More recent versions of FEniCS and mshr may be preferred, but one should be careful of [inconsistent dependencies](http://fenicsproject.discourse.group/t/anaconda-installation-of-fenics-and-mshr/2062/5). If the following code snippet runs successfully, then your environment *should* have been created correctly:
```
from mshr import *
```

## Execution
The ROM-based simulations can be run via `run_rom.py`. The FEM-based simulations can be run via `run_fem.py`.

Code can be run as
```
python3 run_rom.py $example_tag
```
or
```
python3 run_fem.py $example_tag
```
The placeholder `$example_tag` can take the values
* `wedge_1`
* `wedge_2`
* `wedge_3`
* `wedge_4`
* `cavity`
* `room_tol2.5e-2`
* `room_tol1e-3`
* `room_harmonic_1`
* `room_harmonic_5`

Otherwise, one can simply run
```
python3 run_rom.py
```
or
```
python3 run_fem.py
```
and then input `$example_tag` later.

## Acknowledgments
Part of the funding that made this code possible has been provided by the Austrian Science Fund (FWF) through projects F 65 and P 33477.
