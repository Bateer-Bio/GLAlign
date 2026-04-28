# GLAlign: Integrating Single-Cell Multi-Omics Data via Global Manifold and Local Marker Gene Alignment

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-orange.svg)](https://pytorch.org/)

GLAlign (Global manifold and Local marker gene Alignment) is an optimization framework designed to integrate single-cell multi-omics data by jointly leveraging global manifold structure and local marker gene information. It effectively overcomes geometric ambiguity and overcorrection, providing a robust and interpretable framework for accurate single-cell multi-omics integration.

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

## Quick Start

You can run the main alignment script GLAlign.py via the command line. The script requires pre-computed within-modality kernel matrices (K1, K2), a cross-modality similarity matrix (K12), and cell-type labels.
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
