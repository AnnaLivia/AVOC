## AVOC: A Novel Clustering Algorithm with Optimality Guarantees </br>
[Strong bounds for large-scale Minimum Sum-of-Squares Clustering](https://arxiv.org/abs/2502.08397)
<p align="center">
  <img src="Figures/logo.png" width="160" height="200" />
</p>

# 📂 Instances

This folder contains the datasets used in our experiments. We consider both **synthetic** and **real-world** instances.
<br>


## 🧪 Synthetic Instances
To test our algorithm, we generate large-scale Gaussian datasets with:
- **Number of points**: \(N = 10{,}000\)  
- **Dimension**: \(D = 2\)  
- **Number of clusters**: \(K \in \{2, 3, 4\}\)  
- **Noise levels**: \(\sigma \in \{0.50, 0.75, 1.00\}\)  

Each dataset is drawn from a mixture of \(K\) Gaussian distributions with equal proportions and spherical covariance \(\Sigma_j = \sigma I\). Cluster centers \(\mu_j\) are sampled uniformly from \([-10, 10]\).  

- **Naming convention**: `N-K-σ` (e.g., `10000-3-0.75` denotes a dataset with 10,000 points, 3 clusters, and noise level 0.75).  
- For small \(\sigma\), clusters are well separated; higher noise levels make them overlap more.  
<br>

### 🔍 Example Visualizations
Below we show datasets with \(N=10{,}000\), \(K=3\), and \(\sigma \in \{0.50, 0.75, 1.00\}\):

<p align="center">
  <img src="Figures/10000_3_05.png" alt="10000-3-0.50" width="30%">
  <img src="Figures/10000_3_075.png" alt="10000-3-0.75" width="30%">
  <img src="Figures/10000_3_10.png" alt="10000-3-1.00" width="30%">
</p>
<br>


## 🌍 Real-World Instances
We also consider five benchmark datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).  
The number of clusters was selected using the **elbow method**.

| Dataset   | N     | D  | K | Cluster Sizes |
|-----------|-------|----|---|-----------------------------|
| Abalone   | 4,177 | 10 | 3 | 1,308, 1,341, 1,528 |
| Facebook  | 7,050 | 13 | 3 | 218, 2,558, 4,274 |
| Frogs     | 7,195 | 22 | 4 | 605, 670, 2,367, 3,553 |
| Electric  | 10,000| 12 | 3 | 2,886, 3,537, 3,577 |
| Pulsar    | 17,898|  8 | 2 | 2,057, 15,841 |

> ℹ️ The **Abalone** dataset is the largest instance previously solved to optimality by **SOS-SDP**.
