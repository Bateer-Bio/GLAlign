# GLAlign
# GLAlign

GLAlign is a manifold alignment method for single-cell multi-omics data integration. It learns a shared latent space by combining Maximum Mean Discrepancy (MMD), triplet loss with semi-hard negative mining, and regularization for feature and structure preservation.

## Installation

```bash
conda create -n glalign python=3.9
conda activate glalign
pip install -r requirements.txt

Input Files
File	Description
K1.npy	Kernel matrix for dataset 1
K2.npy	Kernel matrix for dataset 2
K12_z.csv	Jaccard similarity matrix between cell types
Label1R.txt	Cell type labels for dataset 1 (one per line)
Label2A.txt	Cell type labels for dataset 2 (one per line)
Usage
bash
python GLAlign.py \
  ./data/PBMC_3CT/K1.npy \
  ./data/PBMC_3CT/K2.npy \
  ./results/ \
  5 0.5 1e-09 1e-07 0.5 \
  ./data/PBMC_3CT/K12_z.csv \
  ./data/PBMC_3CT/Label1R.txt \
  ./data/PBMC_3CT/Label2A.txt

Parameters
Parameter	Description
num_feat	Dimensionality of latent space
sigma	Bandwidth for Gaussian kernel (0 = auto)
lambda1	Weight for penalty term
lambda2	Weight for distortion term
lambda3	Weight for triplet loss
Output
Results are saved under the specified result directory, including:

alpha_hat_<seed>_3000.txt — transformation matrix for dataset 1
beta_hat_<seed>_3000.txt — transformation matrix for dataset 2
Functions_<seed>.png — loss curve plot
all_objectives.txt — summary across seeds
Demo
See Demo.ipynb for a complete example on the PBMC dataset, including data preprocessing, running GLAlign, and UMAP visualization.
