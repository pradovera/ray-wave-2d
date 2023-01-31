# README

Minimal code for running numerical examples in the paper

D. Pradovera, M. Nonino, and I. Perugia, _Geometry-based approximation of waves propagating through complex domains_ (2023)

# Prerequisites
* **numpy** and **scipy**
* **matplotlib**
* **fenics** and **mshr** for 2D FEM tests

# Running
The ROM-based simulations can be run via !!run_rom.py!!.

Code can be run as
```
python3 run_rom $example_tag
```
where `$example_tag` can take the values
* `wedge_1`
* `wedge_2`
* `wedge_3`
* `wedge_4`
* `cavity`
* `room_1e-2`
* `room_1e-3`
* `room_harmonic_1`
* `room_harmonic_5`

## Acknowledgments
Part of the funding that made this code possible has been provided by the Austrian Science Fund (FWF) through projects F 65 and P 33477.
