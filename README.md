## AVOC Algorithm </br>
A Heuristic Solver for Large-scale Minimum Sum-of-Squares Clustering with Optimality Guarantees
<p align="center">
  <img src="https://github.com/AnnaLivia/AVOC/blob/main/logo.png" width="160" height="200" />
</p>

**AVOC** is a heuristic algorithm leveraging the concept of the anticlustering problem—i.e., the problem of partitioning a set of objects into groups with high intra-group dissimilarity and low inter-group dissimilarity.
This repository contains the C++ source code, the MATLAB scripts, and the datasets used for the experiments.

> A. L. Croella, V. Piccialli, A. M. Sudoso, Large-scale Minimum Sum-of-Squares Clustering with Optimality Guarantees, **arxiv** (2025), .


<br>

## Installation
**AVOC** calls [SOS-SDP](https://github.com/antoniosudoso/sos-sdp), an exact algorithm based on the branch-and-bound technique for solving the Minimum Sum-of-Squares Clustering (MSSC), and the semidefinite programming solver [SDPNAL+](https://blog.nus.edu.sg/mattohkc/softwares/sdpnalplus/) by using the [MATLAB Engine API](https://www.mathworks.com/help/matlab/calling-matlab-engine-from-cpp-programs.html) for C++.
It requires the MATLAB engine library *libMatlabEngine* and the Matlab Data Array library *libMatlabDataArray*.

**AVOC** calls the integer programming solver [Gurobi](https://www.gurobi.com/) and uses the [Armadillo](http://arma.sourceforge.net/) library to handle matrices and linear algebra operations efficiently.
Before installing Armadillo, first install OpenBLAS and LAPACK along with the corresponding development files. **AVOC** implements a configurable thread pool of POSIX threads to speed up both the heuristic procedure and the branch-and-bound search.

**AVOC** calls a python script to run a k-means algorithm using [argparse](https://pypi.org/project/argparse/), [pandas](https://pypi.org/project/pandas/) and [scikit-learn](https://pypi.org/project/scikit-learn/) libraries.
Before running the algorithm, first create a virtual env and install argparse, pandas, sklearn.

Ubuntu and Debian instructions:
1) Install MATLAB (>= 2016b)
2) Install Gurobi (>= 9.0)
3) Install CMake, OpenBLAS, LAPACK and Armadillo:
 ```
sudo apt-get update
sudo apt-get install cmake libopenblas-dev liblapack-dev libarmadillo-dev
```
4) Open the makefile `anticlustering_c++/Makefile` 
	- Set the variable `matlab_path` with your MATLAB folder.
	- Set the variable `gurobi_path` with your Gurobi folder.
5) Compile the code:

```
cd anticlustering_c++/
make
```

6) Download SDPNAL+, move the folder `clustering_matlab` containing the MATLAB source code of **AVOC** in the SDPNAL+ main directory and set the parameter `SDP_SOLVER_FOLDER` of the configuration file accordingly. This folder and its subfolders will be automatically added to the MATLAB search path when **AVOC** starts.

7) Install Python (>=3.10)
8) Create a virtual env and install libraries
 ```
sudo apt install python3.10-venv
python3 -m venv myenv
source myenv/bin/activate
pip3 install argparse pandas scikit-learn
```

The code has been tested on macOS system 15.0.1 with MATLAB R2022b, Gurobi c and Armadillo 14.0.

<br>

> [!IMPORTANT]
> You may need to edit the following line in the file _clustering_matlab/solve_cluster_cp.m_ to insert your gurobi path:<br>
 ``` addpath('/your_path_to_gurobi/gurobiXX/linux64/matlab');```

<br>

## Configuration
Various parameters used in **AVOC** can be modified in the configuration file `anticlustering_c++/config.txt`:

1) Result path parameter

- `RESULT_FOLDER ` - path to the results folder

2) AVOC parameters

- `ALGORITHM_MIN_GAP_CRITERION` - algorithm minimum gap lb-ub stop criterion (%)
- `ALGORITHM_MAX_TIME_PER_ANTI_CRITERION` - algorithm maximum time stop criterion (%)
- `ANTICLUSTERING_THREADS` - number of threads for anticlustering
- `PARTITION_THREADS` - number of threads for computing the partitions


3) AVOC evaluation procedure parameters (k-means)

- `KMEANS_NUM_START` -  = number of starts of kmeans
- `KMEANS_MAX_ITERATION` - number of max iterations of kmeans
- `KMEANS_VERBOSE` - do not display kmeans log (0), display log (1)

4) Branch and Bound parameter

- `BRANCH_AND_BOUND_TOL` -  precision of the branch and bound
- `BRANCH_AND_BOUND_PARALLEL` -  single thread (1), multi-thread (> 1)
- `BRANCH_AND_BOUND_MAX_NODES` - max number of the branch and bound nodes (1 - only root)
- `BRANCH_AND_BOUND_VISITING_STRATEGY` - best first (0),  depth first (1), breadth first (2)

5) SOS-SDP parameters

- `BRANCH_AND_BOUND_TOL` - optimality tolerance of the branch-and-bound
- `BRANCH_AND_BOUND_PARALLEL` -  thread pool size: single thread (1), multi-thread (> 1)
- `BRANCH_AND_BOUND_MAX_NODES` - maximum number of nodes
- `BRANCH_AND_BOUND_VISITING_STRATEGY` - best first (0),  depth first (1), breadth first (2)
- `SDP_SOLVER_SESSION_THREADS_ROOT` - number of threads for the MATLAB session at the root
- `SDP_SOLVER_SESSION_THREADS` - number of threads for the MATLAB session for the ML and CL nodes
- `SDP_SOLVER_FOLDER` - full path of the SDPNAL+ folder
- `SDP_SOLVER_TOL` - accuracy of SDPNAL+
- `SDP_SOLVER_VERBOSE` - do not display log (0), display log (1)
- `SDP_SOLVER_MAX_CP_ITER_ROOT` - maximum number of cutting-plane iterations at the root
- `SDP_SOLVER_MAX_CP_ITER` - maximum number of cutting-plane iterations for the ML and CL nodes
- `SDP_SOLVER_CP_TOL` - cutting-plane tolerance between two consecutive cutting-plane iterations
- `SDP_SOLVER_MAX_INEQ` - maximum number of valid inequalities to add
- `SDP_SOLVER_INHERIT_PERC` - fraction of inequalities to inherit
- `SDP_SOLVER_EPS_INEQ` - tolerance for checking the violation of the inequalities
- `SDP_SOLVER_EPS_ACTIVE` - tolerance for detecting the active inequalities
- `SDP_SOLVER_MAX_PAIR_INEQ` - maximum number of pair inequalities to separate
- `SDP_SOLVER_PAIR_PERC` - fraction of the most violated pair inequalities to add
- `SDP_SOLVER_MAX_TRIANGLE_INEQ` - maximum number of triangle inequalities to separate
- `SDP_SOLVER_TRIANGLE_PERC` - fraction of the most violated triangle inequalities to add


<br>

## Usage
```
cd anticlustering_c++/
./bb <DATASET> <ASSIGNMENT> <K> <T>
```
- `DATASET` - path of the dataset file
- `ASSIGNEMENT` - path of the assignemnt file
- `K` - number of clusters
- `T` - number of anticlusters

File `DATASET` contains the data points `x_ij` and the must include an header line with the problem size `n` and the dimension `d`:

```
n d
x_11 x_12 ... x_1d
x_21 x_22 ... x_2d
...
...
x_n1 x_n2 ... x_nd
```

File `ASSIGNEMENT` should include the indices of cluster `k(i)` assigned to data point `k(i)`:

```
k(1)
k(2)
...
...
k(n)
```


<br>

## Log

The log file reports the progress of the AVOC algorithm when a new improvement is found:

- `It` - number of iteration of reapeat cycle
- `k` - number of cluster involved in the swap
- `LB+` - value of the LB+
- `GAP %` - value of the gap between UB and LB+
-  Stopping criterion met


For each anticluster subset created and solved by **SOS-SDP** the file reports:

- `N` - size of the current node
- `NODE_PAR` - id of the parent node
- `NODE` - id of the current node
- `LB_PAR` - lower bound of the parent node
- `LB` - lower bound of the current node
- `FLAG` - termination flag of SDPNAL+
    -  `0` - SDP is solved to the required accuracy
    -  `1` - SDP is not solved successfully
    -  `-1, -2, -3` - SDP is partially solved successfully
- `TIME (s)` - running time in seconds of the current node
- `CP_ITER` - number of cutting-plane iterations
- `CP_FLAG` - termination flag of the cutting-plane procedure
    - `-3` - current bound is worse than the previous one
    - `-2` - SDP is not solved successfully
    - `-1` - maximum number of iterations
    -  `0` - no violated inequalities
    -  `1` - maximum number of inequalities
    -  `2` - node must be pruned
    -  `3` - cutting-plane tolerance
- `CP_INEQ` - number of inequalities added in the last cutting-plane iteration
- `PAIR TRIANGLE CLIQUE` - average number of added cuts for each class of inequalities
- `UB` - current upper bound
- `GUB` - global upper bound
- `I J` - current branching decision
- `NODE_GAP` - gap at the current node
- `GAP` - overall gap 
- `OPEN` - number of open nodes


Log file example:

```
DATA_FILE, n, d, k, t:
../artificial/data/example.txt 64 2 4 4

ALGORITHM_MIN_GAP_CRITERION: 0.001
ALGORITHM_MAX_TIME_PER_ANTI_CRITERION: 240
ANTICLUSTERING_THREADS: 4
PARTITION_THREADS: 4

KMEANS_NUM_START: 100
KMEANS_MAX_ITERATION: 100
KMEANS_VERBOSE: 0

BRANCH_AND_BOUND_TOL: 0.0001
BRANCH_AND_BOUND_PARALLEL: 1
BRANCH_AND_BOUND_MAX_NODES: 1
BRANCH_AND_BOUND_VISITING_STRATEGY: 0

SDP_SOLVER_SESSION_THREADS_ROOT: 2
SDP_SOLVER_SESSION_THREADS: 1
SDP_SOLVER_FOLDER: ../SDPNAL
SDP_SOLVER_TOL: 0.0001
SDP_SOLVER_VERBOSE: 0
SDP_SOLVER_MAX_CP_ITER_ROOT: 80
SDP_SOLVER_MAX_CP_ITER: 40
SDP_SOLVER_CP_TOL: 0.0001
SDP_SOLVER_MAX_INEQ: 100000
SDP_SOLVER_INHERIT_PERC: 1
SDP_SOLVER_EPS_INEQ: 0.0001
SDP_SOLVER_EPS_ACTIVE: 1e-06
SDP_SOLVER_MAX_PAIR_INEQ: 100000
SDP_SOLVER_PAIR_PERC: 0.05
SDP_SOLVER_MAX_TRIANGLE_INEQ: 100000
SDP_SOLVER_TRIANGLE_PERC: 0.1


---------------------------------------------------------------------
UB: 575.67


AVOC Algorithm

It | k | LB+ | GAP %
0 | 0 | 501.37 | 12.9075
1 | 1 | 502.69 | 12.6782
1 | 1 | 521.22 | 9.4599
1 | 1 | 521.89 | 9.3424
1 | 2 | 525.00 | 8.8025
1 | 2 | 530.73 | 7.8076
1 | 2 | 531.47 | 7.6782
1 | 3 | 533.04 | 7.4052
1 | 3 | 533.93 | 7.2506
1 | 3 | 534.73 | 7.1127
1 | 4 | 539.52 | 6.2805
1 | 4 | 542.90 | 5.6925
1 | 4 | 544.76 | 5.3698
1 | 4 | 545.03 | 5.3239
2 | 1 | 548.48 | 4.7242
2 | 1 | 549.86 | 4.4845
2 | 1 | 550.97 | 4.2906
2 | 2 | 551.55 | 4.1900
2 | 2 | 554.84 | 3.6196
2 | 3 | 555.52 | 3.5012
2 | 3 | 556.20 | 3.3828
2 | 4 | 559.21 | 2.8603
2 | 4 | 559.41 | 2.8252
2 | 4 | 562.60 | 2.2715
2 | 4 | 562.74 | 2.2463
3 | 1 | 564.81 | 1.8871
3 | 1 | 568.56 | 1.2356
3 | 1 | 569.24 | 1.1172
3 | 2 | 570.30 | 0.9334
3 | 4 | 572.73 | 0.5113
3 | 4 | 573.47 | 0.3821
4 | 2 | 573.68 | 0.3470
4 | 2 | 573.88 | 0.3124
4 | 4 | 574.30 | 0.2389
5 | 2 | 574.99 | 0.1195
5 | 4 | 575.67 | 0.0000

Min GAP 0.001 reached

---------------------------------------------------------------------

PARTITION 1

|    N| NODE_PAR|    NODE|      LB_PAR|          LB|  FLAG|  TIME (s)| CP_ITER| CP_FLAG|   CP_INEQ|     PAIR  TRIANGLE    CLIQUE|          UB|         GUB|     I      J| NODE_GAP (%)|      GAP (%)|  OPEN|
|   16|       -1|       0|        -inf|     143.882|     0|     0.000|       0|       2|         0|    0.000     0.000     0.000|     143.883|    143.883*|    -1     -1|        0.000|        0.000|     0|

WALL_TIME: 2 sec
N_NODES: 1
AVG_INEQ: 0.000
AVG_CP_ITER: 0.000
ROOT_GAP: 0.000
GAP: 0.000
ROOT_LB: 143.882
BEST_UB: 143.883
BEST_LB: 143.882


PARTITION 4
|    N| NODE_PAR|    NODE|      LB_PAR|          LB|  FLAG|  TIME (s)| CP_ITER| CP_FLAG|   CP_INEQ|     PAIR  TRIANGLE    CLIQUE|          UB|         GUB|     I      J| NODE_GAP (%)|      GAP (%)|  OPEN|
|   16|       -1|       0|        -inf|     143.888|     0|     0.000|       0|       0|         0|    0.000     0.000     0.000|     143.922|    143.922*|    -1     -1|        0.024|        0.024|     0|
PRUNING BY OPTIMALITY 0

WALL_TIME: 2 sec
N_NODES: 1
AVG_INEQ: 0.000
AVG_CP_ITER: 0.000
ROOT_GAP: 0.000
GAP: 0.000
ROOT_LB: 143.888
BEST_UB: 143.922
BEST_LB: 143.888


PARTITION 2
|    N| NODE_PAR|    NODE|      LB_PAR|          LB|  FLAG|  TIME (s)| CP_ITER| CP_FLAG|   CP_INEQ|     PAIR  TRIANGLE    CLIQUE|          UB|         GUB|     I      J| NODE_GAP (%)|      GAP (%)|  OPEN|
|   16|       -1|       0|        -inf|     143.896|     0|     0.000|       0|       2|         0|    0.000     0.000     0.000|     143.896|    143.896*|    -1     -1|        0.000|        0.000|     0|

WALL_TIME: 2 sec
N_NODES: 1
AVG_INEQ: 0.000
AVG_CP_ITER: 0.000
ROOT_GAP: 0.000
GAP: 0.000
ROOT_LB: 143.896
BEST_UB: 143.896
BEST_LB: 143.896


PARTITION 3
|    N| NODE_PAR|    NODE|      LB_PAR|          LB|  FLAG|  TIME (s)| CP_ITER| CP_FLAG|   CP_INEQ|     PAIR  TRIANGLE    CLIQUE|          UB|         GUB|     I      J| NODE_GAP (%)|      GAP (%)|  OPEN|
|   16|       -1|       0|        -inf|     143.972|     0|     0.000|       0|       2|         0|    0.000     0.000     0.000|     143.972|    143.972*|    -1     -1|        0.000|        0.000|     0|

WALL_TIME: 1 sec
N_NODES: 1
AVG_INEQ: 0.000
AVG_CP_ITER: 0.000
ROOT_GAP: 0.000
GAP: 0.000
ROOT_LB: 143.972
BEST_UB: 143.972
BEST_LB: 143.972


*********************************************************************
LB: 575.638
LB+: 575.674
ANTI OBJ: 575.674
Init sol: 575.674
SWAPS: 35
STOP: tol
GAP UB-LB 0.006
GAP UB-LB+ 0.000
*********************************************************************


```


## Related Works

> V. Piccialli, A. Russo Russo, A. M. Sudoso, PC-SOS-SDP: An Exact Algorithm for Semi-supervised Minimum Sum-of-Squares Clustering, **Computers & Operations Research** (2022)
> - Paper: https://doi.org/10.1016/j.cor.2022.105958
> - Code: https://github.com/antoniosudoso/pc-sos-sdp


<br>

> V. Piccialli, A. M. Sudoso, A. Wiegele, SOS-SDP: an Exact Solver for Minimum Sum-of-Squares Clustering, **INFORMS Journal on Computing** (2022).
> - Paper: https://doi.org/10.1287/ijoc.2022.1166
> - Code: https://github.com/antoniosudoso/sos-sdp


<br>

> A. L. Croella: Anticlustering for Large Scale Clustering, General Conference FAIR, Naples 23-24/09/2024
> - Poster: [https://PosterFAIR](https://uniroma1it-my.sharepoint.com/:b:/g/personal/croella_1544694_studenti_uniroma1_it/EScY_IIbJqtIt2BU7NrFvUIBZXxXX-1DVnxqn75ATRx3uw?e=LmZMhS)


<br>

## ﻿Acknowledgements

The work of Anna Livia Croella and Veronica Piccialli is supported by the FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, investment 1.3, line on Artificial Intelligence).
