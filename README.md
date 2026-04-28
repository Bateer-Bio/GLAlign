# GLAlign: Integrating Single-Cell Multi-Omics Data via Global Manifold and Local Marker Gene Alignment

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-orange.svg)](https://pytorch.org/)

GLAlign (Global manifold and Local marker gene Alignment) is an optimization framework designed to integrate single-cell multi-omics data by jointly leveraging global manifold structure and local marker gene information. It effectively overcomes geometric ambiguity and overcorrection, providing a robust and interpretable framework for accurate single-cell multi-omics integration.
<img width="1599" height="766" alt="截屏2026-04-28 15 01 47" src="https://github.com/user-attachments/assets/4ca65cf8-0b57-4a27-9e10-e162bbb26494" />
<img width="1085" height="283" alt="截屏2026-04-28 15 02 13" src="https://github.com/user-attachments/assets/0fa37c1e-01b2-42dc-a657-dc2170c177b8" />
## Overview
GLAlign learns a shared latent space using a joint objective that combines:
1. **Maximum Mean Discrepancy (MMD)** for global distribution matching.
2. **Distortion and Penalty terms** for structural and feature preservation.
3. **Contrastive Triplet Loss** guided by cross-modality marker gene similarity for local biological consistency.

## Installation

We recommend using `conda` to create a virtual environment for GLAlign. The model is implemented in PyTorch.

```bash
# Create and activate a new conda environment
conda create -n glalign-env python=3.12
conda activate glalign-env

# Install required packages
pip install torch numpy pandas matplotlib scanpy seaborn scib-metrics
```
## **Quick Start**

### 1. Running GLAlign

You can run the main alignment script `GLAlign.py` via the command line.
The script requires pre-computed within-modality kernel matrices (`K1`, `K2`), a cross-modality similarity matrix (`K12`), and cell-type labels.

#### Example Command:
```bash
python GLAlign.py \
    ./K1.npy \
    ./K2.npy \
    ./MMDMAT_result/ \
    5 \
    0.5 \
    1e-09 \
    1e-07 \
    0.5 \
    ./K12_z.csv \
    ./Label1R.txt \
    ./Label2A.txt
```
Arguments Explanation:
- **​​input_k1​​:** Path to the kernel matrix for dataset 1 (e.g., K1.npy).
- **input_k2:** Path to the kernel matrix for dataset 2 (e.g., K2.npy).  
- **​​result_dir​​:** Directory to save the output transformation matrices and loss plots.
- **num_feat:** Dimensionality of the shared latent space (e.g., 5).
- **sigma​​:** Bandwidth parameter for the Gaussian kernel (e.g., 0.5).
- **lambda1:** Weight for the penalty term (e.g., 1e-09).
- **lambda2:** Weight for the distortion term (e.g., 1e-07).
- **lambda3​​:** Weight for the contrastive triplet loss term (e.g., 0.5).
- **k12:** Path to the CSV file with the Jaccard similarity matrix (K12_z.csv).
- **label1​​:** Path to cell type labels for dataset 1.
- **label2:** Path to cell type labels for dataset 2. 

### 2. Downstream Analysis & Visualization
After running GLAlign.py, you can load the learned transformation matrices (alpha_hat and beta_hat) into Python to compute the integrated embeddings and visualize them using scanpy.
```bash
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# 1. Load original kernel matrices
K1 = np.load("./K1.npy")
K2 = np.load("./K2.npy")

# 2. Load the learned transformation matrices from GLAlign output
# Note: Adjust the path based on your specified result_dir and best seed
alpha = pd.read_csv('./MMDMAT_result/results_nfeat_5_sigma_0.5_lam1_1e-09_lam2_1e-07_lam3_0.5/seed_5/alpha_hat_5_3000.txt', sep=r'\s+', header=None)
beta = pd.read_csv('./MMDMAT_result/results_nfeat_5_sigma_0.5_lam1_1e-09_lam2_1e-07_lam3_0.5/seed_5/beta_hat_5_3000.txt', sep=r'\s+', header=None)

# 3. Compute the shared latent embeddings
Z1 = np.dot(K1, alpha.values)
Z2 = np.dot(K2, beta.values)
MMDMAT_emb = np.vstack((Z1, Z2))

# 4. Add embeddings to your AnnData object
# Assuming `adata` is your combined AnnData object
adata.obsm['GLAlign'] = MMDMAT_emb

# 5. Compute neighbors and UMAP
sc.pp.neighbors(adata, use_rep='GLAlign')
sc.tl.umap(adata, random_state=666)

# 6. Plot UMAP
sc.pl.umap(adata, color=["celltype"], palette={'CD8 Naive':"#2ca02c", 'NK dim':"#d62728", 'CD16+ Monocytes':"#9467bd"}, save="_CT_M3Align.pdf")
sc.pl.umap(adata, color=["batch"], save="_batch_M3Align.pdf")
```
<img width="644" height="414" alt="截屏2026-04-28 15 02 55" src="https://github.com/user-attachments/assets/b6616e75-ded5-4f6d-a86b-eb297fba1704" />
<img width="685" height="414" alt="截屏2026-04-28 15 02 44" src="https://github.com/user-attachments/assets/78de3910-7c78-4bc9-9c42-f833c2b67bdf" />







