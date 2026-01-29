# Rod Climbing of Viscoelastic Models

This repository provides a fully runnable minimal numerical experiment accompanying the paper
*A thermodynamically consistent Johnson--Segalman--Giesekus model: numerical simulation of the rod climbing effect*. The implementation is built on the **Firedrake** finite
element library and features a time-stepping solver for **Johnson–Segalman–Giesekus-type**
viscoelastic models.

At present, the code is configured to reproduce results for the **Oldroyd-B model** over
a range of rod rotation speeds.

The code is released under the **MIT License**.

## Requirements

- **Firedrake** (environment specified in `firedrake.json`)
- **Netgen** (ngsPETSc)

## Usage

Run the script in parallel with MPI:

`mpiexec -n 8 python rod_climbing_code.py`

or in serial

`python rod_climbing_code.py`

## Results

The solver reproduces the results reported in the **Oldroyd-B** section of the paper and saves
them to the directory `results_Oldroyd`.

All `data` used in the figures are provided in the data directory, together with scripts for
post-processing and plotting, organized according to the corresponding subsections of the paper.

## Performance

Parallel execution with `mpiexec -n 8` results in a runtime of approximately 10 minutes for one
high-accuracy setup on an average laptop. The first run on a given machine includes an initial
form-assembly overhead of about 4 minutes. Using `-- fe_polynomial_degree = 2` reduces the runtime
to approximately 4 minutes, while maintaining accuracy comparable to the high-accuracy solution.
