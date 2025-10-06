# AVOC: A Novel Clustering Algorithm with Optimality Guarantees </br>

<p align="center">
  <img src="https://github.com/AnnaLivia/AVOC/blob/main/Figures/logo.png" width="160" height="200" />
</p>

This repository contains the **full source code and benchmark datasets** used in the paper:  

> A. L. Croella, V. Piccialli, A. M. Sudoso, Strong bounds for large-scale Minimum Sum-of-Squares Clustering, **[arXiv](https://arxiv.org/abs/2502.08397)** (2025)
---

## 🧩 Overview
This repository provides everything required to **reproduce the experiments** reported in the paper.

The benchmark instances include both **synthetic datasets** and **real-world datasets** from the UCI repository (see [`Instances/`](./Instances/) and its `README.md` for details).

<br>

## 🛠️ Installation
**AVOC** calls [SOS-SDP](https://github.com/antoniosudoso/sos-sdp), an exact algorithm based on branch-and-bound for solving the Minimum Sum-of-Squares Clustering (MSSC), and the semidefinite programming solver [SDPNAL+](https://blog.nus.edu.sg/mattohkc/softwares/sdpnalplus/) via the [MATLAB Engine API](https://www.mathworks.com/help/matlab/calling-matlab-engine-from-cpp-programs.html) for C++.  
It requires the MATLAB engine library *libMatlabEngine* and the Matlab Data Array library *libMatlabDataArray*.

**AVOC** also:
- calls the integer programming solver [Gurobi](https://www.gurobi.com/)  
- uses [Armadillo](http://arma.sourceforge.net/) for matrix and linear algebra operations  
- implements a thread pool with POSIX threads for speedup  
- calls a Python script for k-means using `argparse`, `pandas`, and `scikit-learn`

### Ubuntu / Debian Instructions
1. Install MATLAB (>= 2016b)  
2. Install Gurobi (>= 9.0)  
3. Install dependencies:
   ```bash
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

> [!IMPORTANT]
> You may need to edit the following line in the file _clustering_matlab/solve_cluster_cp.m_ to insert your gurobi path:<br>
 ``` addpath('/your_path_to_gurobi/gurobiXX/your_operating_system/matlab');```
> 
<br>

## ⚙️ Configuration
Parameters used in **AVOC** can be set in the configuration file [`anticlustering_c++/config.txt`](./anticlustering_c++/config.txt).
They include AVOC tolerances, k-means iterations, branch-and-bound settings, SOS-SDP solver parameters, and thread configurations (see file comments for details).

<br>

## ▶️ Usage
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

## 📑 Log Output

The log file reports AVOC’s progress when improvements are found, including:
- `It` - iteration number
- `k` - cluster involved in the swap
- `LB+` - value of the LB+
- `GAP %` - gap between UB and LB+
-  Stopping criterion met

It also contains detailed SOS-SDP information for each anticluster (bounds, flags, cuts, runtime, etc.).

👉 See the [`example`](Results/example/example_4_LOG.txt) for details.

---
<br>

## 📚 Related Works

> V. Piccialli, A. Russo Russo, A. M. Sudoso, PC-SOS-SDP: An Exact Algorithm for Semi-supervised Minimum Sum-of-Squares Clustering, **Computers & Operations Research** (2022)
> - Paper: https://doi.org/10.1016/j.cor.2022.105958
> - Code: https://github.com/antoniosudoso/pc-sos-sdp

> V. Piccialli, A. M. Sudoso, A. Wiegele, SOS-SDP: an Exact Solver for Minimum Sum-of-Squares Clustering, **INFORMS Journal on Computing** (2022).
> - Paper: https://doi.org/10.1287/ijoc.2022.1166
> - Code: https://github.com/antoniosudoso/sos-sdp

> A. L. Croella: Anticlustering for Large Scale Clustering, General Conference FAIR, Naples 23-24/09/2024
> - Poster: [https://PosterFAIR](https://uniroma1it-my.sharepoint.com/:b:/g/personal/croella_1544694_studenti_uniroma1_it/EScY_IIbJqtIt2BU7NrFvUIBZXxXX-1DVnxqn75ATRx3uw?e=LmZMhS)
<br>

## ✍️ How to Cite
If you use **AVOC** in your work, please cite:

```bibtex
@article{croella2025avoc,
  title     = {Strong bounds for large-scale Minimum Sum-of-Squares Clustering},
  author    = {Croella, Anna Livia and Piccialli, Veronica and Sudoso, Antonio M.},
  journal   = {arXiv preprint arXiv:2502.08397},
  year      = {2025}
}
```
<br>

## 🙏 Acknowledgements

The work of Anna Livia Croella and Veronica Piccialli has been supported by the FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, investment 1.3, line on Artificial Intelligence).
<br>