"""
Manifold Alignment with Maximum Mean Discrepancy (MMD) and Triplet Loss

This script implements a manifold alignment technique that combines:
1. Maximum Mean Discrepancy (MMD) for distribution matching
2. Triplet loss with semi-hard negative mining for structural preservation
3. Regularization terms for feature preservation

The method aligns two datasets (k1 and k2) by learning transformation matrices
that project both datasets into a shared latent space while preserving:
- Global distribution similarity (via MMD)
- Local neighborhood structures (via triplet loss)
- Feature space properties (via regularization terms)

Usage:
python manifold_align_mmd_triplet.py <input_k1> <input_k2> <result_dir> <num_feat> <sigma> 
        <lambda1> <lambda2> <lambda3> <k12> <label1> <label2>

Arguments:
  input_k1    : Path to kernel matrix for dataset 1 (numpy .npy file)
  input_k2    : Path to kernel matrix for dataset 2 (numpy .npy file)
  result_dir  : Directory to save results
  num_feat    : Dimensionality of latent space
  sigma       : Bandwidth parameter for Gaussian kernel
  lambda1     : Weight for penalty term
  lambda2     : Weight for distortion term
  lambda3     : Weight for triplet loss term
  k12         : Path to CSV file with Jaccard similarity matrix
  label1      : Path to cell type labels for dataset 1
  label2      : Path to cell type labels for dataset 2
"""

import numpy as np
import math
import sys
import os
import matplotlib
import pandas as pd

# Set matplotlib backend to Agg for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.cuda
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# Validate command line arguments
USAGE = """USAGE: python manifold_align_mmd_triplet.py 
<input_k1> <input_k2>  <result_dir> <num_feat> <sigma> 
<lambda1> <lambda2> <lambda3> <k12> <label1> <label2>"""

if len(sys.argv) < 12:
    sys.stderr.write(USAGE)
    sys.exit(1)

try:
    nfeat = int(sys.argv[4])
    sigma = float(sys.argv[5])
    lambda_1 = float(sys.argv[6])
    lambda_2 = float(sys.argv[7])
    lambda_3 = float(sys.argv[8])
except ValueError:
    sys.stderr.write("Error: num_feat, sigma, lambda1, lambda2, lambda3 should be numeric values.\n")
    sys.exit(1)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


def compute_pairwise_distances(x, y):
    """
    Compute pairwise squared Euclidean distances between two sets of points
    
    Args:
        x: Tensor of shape (n, d)
        y: Tensor of shape (m, d)
    
    Returns:
        Tensor of shape (n, m) with pairwise squared distances
    """
    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if list(x.size())[1] != list(y.size())[1]:
        raise ValueError('The number of features should be the same.')

    # Efficient computation using broadcasting
    diff = (x.unsqueeze(2) - y.t())
    diff = (diff ** 2).sum(1)
    return diff.t()


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Compute Gaussian kernel matrix between two sets of points
    
    Args:
        x: Tensor of shape (n, d)
        y: Tensor of shape (m, d)
        sigmas: Tensor of bandwidth parameters
    
    Returns:
        Kernel matrix of shape (n, m)
    """
    beta = 1.0 / (2.0 * (sigmas.unsqueeze(1)))
    dist = compute_pairwise_distances(x, y)
    s = beta * (dist.contiguous()).view(1, -1)
    result = ((-s).exp()).sum(0)
    return (result.contiguous()).view(dist.size())


def compute_mmd(x, y, sigmas, kernel=gaussian_kernel_matrix):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    
    Args:
        x: Samples from first distribution
        y: Samples from second distribution
        sigmas: Bandwidth parameters for Gaussian kernel
        kernel: Kernel function to use
    
    Returns:
        MMD value
    """
    cost = (kernel(x, x, sigmas)).mean()
    cost += (kernel(y, y, sigmas)).mean()
    cost -= 2.0 * (kernel(x, y, sigmas)).mean()

    # Ensure non-negative MMD
    if cost.data.item() < 0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost


def calculate_sigma(x1, x2):
    """
    Automatically calculate bandwidth parameter for Gaussian kernel
    
    Args:
        x1: Samples from first distribution (numpy array)
        x2: Samples from second distribution (numpy array)
    
    Returns:
        Computed sigma value
    """
    const = 8
    mat = np.concatenate((x1, x2))
    dist = []
    nsamp = mat.shape[0]
    
    # Compute median distance to nearest neighbor
    for i in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(mat[i, :], mat)), axis=1))
        dist.append(sorted(euc_dist)[1])
    
    sigma = np.square(const * np.median(dist))
    print(f"Calculated sigma: {sigma}")
    return sigma


class CosineTripletLoss(nn.Module):
    """
    Cosine similarity-based triplet loss
    
    Args:
        margin: Minimum margin between positive and negative pairs
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor samples
            positive: Positive samples
            negative: Negative samples
        
        Returns:
            Triplet loss value
        """
        cos_sim_pos = F.cosine_similarity(anchor, positive)
        cos_sim_neg = F.cosine_similarity(anchor, negative)
        cos_dist_pos = 1 - cos_sim_pos
        cos_dist_neg = 1 - cos_sim_neg
        loss = torch.mean(torch.maximum(cos_dist_pos - cos_dist_neg + self.margin, 
                                        torch.tensor(0.0).to(anchor.device)))
        return loss


def select_triplets(matrix1, matrix2, k12_df, label1_types, label2_types, k=0.8):
    """
    Select triplets for triplet loss using semi-hard negative mining
    
    Strategy:
    - Anchor: Cells from dataset 1
    - Positive: Cells from dataset 2 with highest Jaccard similarity
    - Negative: Cells from dataset 2 with similarity < max_similarity * k
    
    Args:
        matrix1: Embeddings for dataset 1
        matrix2: Embeddings for dataset 2
        k12_df: DataFrame with Jaccard similarity matrix
        label1_types: Cell type labels for dataset 1
        label2_types: Cell type labels for dataset 2
        k: Threshold coefficient for negative selection
    
    Returns:
        anchors: Anchor samples
        positives: Positive samples
        negatives: Negative samples
        triplet_info: Information about selected triplets
    """
    # Get cell type names from similarity matrix
    label1_names = k12_df.index.values
    label2_names = k12_df.columns.values
    similarity_matrix = k12_df.values.astype(np.float32)
    
    # Create mapping from cell type to indices
    label1_type_to_indices = {}
    for cell_type in np.unique(label1_types):
        label1_type_to_indices[cell_type] = np.where(label1_types == cell_type)[0]
    
    label2_type_to_indices = {}
    for cell_type in np.unique(label2_types):
        label2_type_to_indices[cell_type] = np.where(label2_types == cell_type)[0]
    
    # Collect all triplets
    all_anchors = []
    all_positives = []
    all_negatives = []
    triplet_info = []  # Store triplet type information
    
    # Iterate over each cell type in dataset 1
    for label1_type in np.unique(label1_types):
        # Skip if cell type not in similarity matrix
        if label1_type not in label1_names:
            print(f"Warning: label1 type '{label1_type}' not in k12 matrix index")
            continue
            
        row_idx = np.where(label1_names == label1_type)[0][0]
        similarities = similarity_matrix[row_idx, :]
        
        # Find most similar cell type in dataset 2
        max_sim = np.max(similarities)
        max_idx = np.argmax(similarities)
        pos_label2_type = label2_names[max_idx]
        
        # Skip if no positive type found
        if max_sim <= 0:
            print(f"Warning: label1 type '{label1_type}' has no positive type")
            continue
        
        # Determine negative types (similarity < max_sim * k)
        neg_threshold = max_sim * k
        neg_label2_types = label2_names[similarities < neg_threshold]
        
        # Skip if no negative types
        if len(neg_label2_types) == 0:
            print(f"Warning: label1 type '{label1_type}' has no negative types")
            continue
        
        # Get indices for current cell type
        label1_indices = label1_type_to_indices[label1_type]
        
        # Get indices for positive cell type
        if pos_label2_type not in label2_type_to_indices:
            print(f"Warning: label2 type '{pos_label2_type}' not in dataset")
            continue
        pos_label2_indices = label2_type_to_indices[pos_label2_type]
        
        # Determine sample size (min of both types)
        num_samples = min(len(label1_indices), len(pos_label2_indices))
        if num_samples == 0:
            continue
            
        # Randomly select anchors and positives
        anchor_idx = np.random.choice(label1_indices, num_samples, replace=False)
        pos_idx = np.random.choice(pos_label2_indices, num_samples, replace=False)
        
        # Select negatives for each sample
        negatives = []
        neg_types = []
        for i in range(num_samples):
            # Randomly select a negative type
            neg_label2_type = np.random.choice(neg_label2_types)
            
            # Ensure selected type exists
            while neg_label2_type not in label2_type_to_indices:
                neg_label2_type = np.random.choice(neg_label2_types)
                
            neg_label2_indices = label2_type_to_indices[neg_label2_type]
            neg_idx = np.random.choice(neg_label2_indices, 1)[0]
            negatives.append(neg_idx)
            neg_types.append(neg_label2_type)
        
        # Normalize embeddings
        anchors = F.normalize(matrix1[anchor_idx], p=2, dim=1)
        positives = F.normalize(matrix2[pos_idx], p=2, dim=1)
        negatives_tensor = F.normalize(matrix2[negatives], p=2, dim=1)
        
        # Add to collections
        all_anchors.append(anchors)
        all_positives.append(positives)
        all_negatives.append(negatives_tensor)
        
        # Record triplet information
        for i in range(num_samples):
            triplet_info.append((label1_type, pos_label2_type, neg_types[i]))
    
    # Return None if no triplets found
    if len(all_anchors) == 0:
        return None, None, None, None
    
    return (torch.cat(all_anchors), 
            torch.cat(all_positives), 
            torch.cat(all_negatives),
            triplet_info)


class ManifoldAlignment(nn.Module):
    """
    Manifold Alignment Model
    
    Learns transformation matrices to align two datasets in a shared latent space
    using MMD, triplet loss, and regularization terms.
    
    Args:
        nfeat: Dimensionality of latent space
        num_k1: Number of samples in dataset 1
        num_k2: Number of samples in dataset 2
        margin: Margin for triplet loss
    """
    def __init__(self, nfeat, num_k1, num_k2, margin=0.2):
        super(ManifoldAlignment, self).__init__()
        # Initialize transformation matrices
        self.alpha = Parameter(torch.FloatTensor(num_k1, nfeat).uniform_(0.0, 0.1).to(device))
        self.beta = Parameter(torch.FloatTensor(num_k2, nfeat).uniform_(0.0, 0.1).to(device))
        self.triplet_loss = CosineTripletLoss(margin=margin)
        
    def forward(self, k1, k2, ip, sigmas, lambda1, lambda2, lambda3, k12_df, label1_types, label2_types):
        """
        Forward pass with loss computation
        
        Args:
            k1: Kernel matrix for dataset 1
            k2: Kernel matrix for dataset 2
            ip: Identity matrix for regularization
            sigmas: Bandwidth parameters for MMD
            lambda1: Weight for penalty term
            lambda2: Weight for distortion term
            lambda3: Weight for triplet loss
            k12_df: Jaccard similarity matrix
            label1_types: Cell type labels for dataset 1
            label2_types: Cell type labels for dataset 2
        
        Returns:
            mmd: MMD loss
            penalty: Regularization term
            distortion: Reconstruction term
            triplet: Triplet loss
            sigmas: Updated bandwidth parameters
            triplet_info: Information about selected triplets
        """
        # Automatically compute sigma if needed
        if sigmas == 0:
            x1 = (torch.matmul(k1, self.alpha)).detach().cpu().numpy()
            x2 = (torch.matmul(k2, self.beta)).detach().cpu().numpy()
            sigma = calculate_sigma(x1, x2)
            sigmas = torch.FloatTensor([sigma]).to(device)

        # Project datasets into latent space
        z_label1 = torch.matmul(k1, self.alpha)
        z_label2 = torch.matmul(k2, self.beta)

        # Compute MMD between latent representations
        mmd = compute_mmd(z_label1, z_label2, sigmas)

        # Compute penalty term (feature preservation)
        penalty = lambda1 * ((torch.matmul(self.alpha.t(), torch.matmul(k1, self.alpha)) - ip).norm(2)
                             + (torch.matmul(self.beta.t(), torch.matmul(k2, self.beta)) - ip).norm(2))

        # Compute distortion term (structure preservation)
        distortion = lambda2 * ((torch.matmul(z_label1, z_label1.t()) - k1).norm(2) +
                                (torch.matmul(z_label2, z_label2.t()) - k2).norm(2))

        # Select triplets and compute triplet loss
        anchors, positives, negatives, triplet_info = select_triplets(
            z_label1, z_label2, k12_df, label1_types, label2_types, 0.8)
        
        if anchors is None:
            print("Warning: No triplets selected in this iteration")
            triplet_value = torch.FloatTensor([0.0]).to(device)
        else:
            triplet_value = self.triplet_loss(anchors, positives, negatives)
        
        triplet = lambda3 * triplet_value
        
        return mmd, penalty, distortion, triplet, sigmas, triplet_info


def plot_data(filename, k, i, obj, mmd, pen, dist, tri, nfeat, sigma, lambda1, lambda2, lambda3):
    """
    Plot loss components over iterations
    
    Args:
        filename: Output file path
        k: Seed value
        i: Current iteration
        obj: Objective values
        mmd: MMD values
        pen: Penalty values
        dist: Distortion values
        tri: Triplet loss values
        nfeat: Latent dimension
        sigma: Bandwidth parameter
        lambda1: Penalty weight
        lambda2: Distortion weight
        lambda3: Triplet weight
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel('Iteration')
    plt.ylabel('log(Function value)')
    plt.title(f'nfeat:{nfeat}, seed:{k}, sigma:{sigma}, lambda1:{lambda1}, lambda2:{lambda2}, lambda3:{lambda3}', 
              fontsize=10)

    plt.plot(obj, 'k--', label='Objective')
    plt.plot(mmd, 'r--', label='MMD')
    plt.plot(pen, 'b--', label='Penalty')
    plt.plot(dist, 'g--', label='Distortion')
    plt.plot(tri, 'm--', label='Triplet')

    if i == 3000:
        plt.legend(loc='upper right', fontsize=8)
    plt.savefig(filename)
    plt.close()


def plot_total_loss(filename, total_losses, nfeat, sigma, lambda1, lambda2, lambda3):
    """
    Plot total loss over iterations
    
    Args:
        filename: Output file path
        total_losses: List of total loss values
        nfeat: Latent dimension
        sigma: Bandwidth parameter
        lambda1: Penalty weight
        lambda2: Distortion weight
        lambda3: Triplet weight
    """
    plt.figure(figsize=(10, 6))
    plt.xlabel('Iteration')
    plt.ylabel('Total Loss')
    plt.title(f'Total Loss: nfeat:{nfeat}, sigma:{sigma}, lambda1:{lambda1}, lambda2:{lambda2}, lambda3:{lambda3}', 
              fontsize=10)
    
    plt.plot(total_losses, 'b-', label='Total Loss')
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(filename)
    plt.close()


def main():
    """Main training procedure"""
    # Set random seeds for reproducibility
    base_seed = 42
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load input data
    print("Loading data...")
    k1_matrix = np.load(sys.argv[1]).astype(np.float32)
    k2_matrix = np.load(sys.argv[2]).astype(np.float32)
    print(f"Dataset 1 shape: {k1_matrix.shape}")
    print(f"Dataset 2 shape: {k2_matrix.shape}")

    # Get dataset sizes
    num_k1 = k1_matrix.shape[0]
    num_k2 = k2_matrix.shape[0]
    nfeat = int(sys.argv[4])
    print(f"Latent space dimensions: {nfeat}")
    
    # Parse parameters
    sigma = float(sys.argv[5])
    sigmas = torch.FloatTensor([sigma]).to(device)
    lambda_1 = float(sys.argv[6])
    lambda_2 = float(sys.argv[7])
    lambda_3 = float(sys.argv[8])
    k12_df = pd.read_csv(sys.argv[9], index_col=0)
    
    # Load cell type labels
    with open(sys.argv[10], "r") as f:
        label1_types = np.array([line.strip() for line in f])

    with open(sys.argv[11], "r") as f:
        label2_types = np.array([line.strip() for line in f])

    # Create results directory
    results_dir = sys.argv[3]
    if not results_dir.endswith('/'):
        results_dir += '/'
        
    base_results_dir = results_dir + f"results_nfeat_{nfeat}_sigma_{sigma}_lam1_{lambda_1}_lam2_{lambda_2}_lam3_{lambda_3}/"
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)

    # Prepare identity matrix for regularization
    Ip = np.identity(nfeat).astype(np.float32)
    K1 = torch.from_numpy(k1_matrix).to(device)
    K2 = torch.from_numpy(k2_matrix).to(device)
    I_p = torch.from_numpy(Ip).to(device)

    # Store objectives across seeds
    all_seeds_objectives = []

    # Run with multiple random seeds
    for seed in range(10):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create seed-specific results directory
        results_dir_seed = base_results_dir + f"seed_{seed}/"
        if not os.path.exists(results_dir_seed):
            os.makedirs(results_dir_seed)

        # Initialize logging containers
        obj_val = []
        mmd_val = []
        pen_val = []
        dist_val = []
        tri_val = []
        total_losses = []
        last_triplet_info = None

        # Initialize model and optimizer
        model = ManifoldAlignment(nfeat, num_k1, num_k2)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, amsgrad=True)
        model.train()

        # Training loop
        for i in range(3001):
            optimizer.zero_grad()
            
            # Forward pass
            mmd, penalty, distortion, triplet, sigmas, triplet_info = model(
                K1, K2, I_p, sigmas, lambda_1, lambda_2, lambda_3, 
                k12_df, label1_types, label2_types)
            
            # Compute total loss
            obj = mmd + penalty + distortion + triplet

            # Logging
            num_triplets = len(triplet_info) if triplet_info is not None else 0
            if i % 3000 == 0:
                print(f"Iter {i:04d} | "
                      f"MMD: {mmd.item():.4f} | "
                      f"Penalty: {penalty.item():.4f} | "
                      f"Distortion: {distortion.item():.4f} | "
                      f"Triplet Loss: {triplet.item():.4f} | "
                      f"Total Loss: {obj.item():.4f} | "
                      f"Triplets: {num_triplets}")

            # Backpropagation
            obj.backward()
            optimizer.step()

            # Store values for plotting
            obj_value = obj.data.item()
            mmd_value = mmd.data.item()
            pen_value = penalty.data.item()
            dist_value = distortion.data.item()
            tri_value = triplet.data.item()
            
            total_losses.append(obj_value)

            if mmd_value > 0:
                obj_val.append(math.log(obj_value))
                mmd_val.append(math.log(mmd_value))
                pen_val.append(math.log(pen_value))
                dist_val.append(math.log(dist_value))
                tri_val.append(math.log(max(1e-40, tri_value)))
            
            # Save triplet info from last iteration
            if i == 3000 and triplet_info is not None:
                last_triplet_info = triplet_info

            # Periodic saving and plotting
            if i % 200 == 0:
                weights = []

                for p in model.parameters():
                    if p.requires_grad:
                        weights.append(p.data)

                # Plot component losses
                plot_file = results_dir_seed + f"Functions_{seed}.png"
                plot_data(plot_file, seed, i, obj_val, mmd_val, pen_val,
                          dist_val, tri_val, nfeat, sigma, lambda_1, lambda_2, lambda_3)
                
                # Plot total loss
                loss_plot_file = results_dir_seed + f"Total_Loss_{seed}.png"
                plot_total_loss(loss_plot_file, total_losses, nfeat, sigma, lambda_1, lambda_2, lambda_3)
                
                # Save transformation matrices
                if i == 0 or i == 3000:
                    np.savetxt(results_dir_seed + f"alpha_hat_{seed}_{i}.txt",
                               weights[0].cpu().numpy())
                    np.savetxt(results_dir_seed + f"beta_hat_{seed}_{i}.txt",
                               weights[1].cpu().numpy())

        # Save final metrics
        final_metrics = {
            "objective": obj.data.item(),
            "mmd": mmd.data.item(),
            "penalty": penalty.data.item(),
            "distortion": distortion.data.item(),
            "triplet": triplet.data.item()
        }
        np.save(results_dir_seed + "final_metrics.npy", final_metrics)
        
        # Save objective value
        objective_file = base_results_dir + f"objective_{seed}.txt"
        with open(objective_file, 'w') as f:
            f.write(str(obj.data.item()))
        
        # Store for summary
        all_seeds_objectives.append(obj.data.item())
        
        # Save triplet info
        if last_triplet_info:
            with open(results_dir_seed + f"triplet_info_{seed}.txt", "w") as f:
                f.write("anchor_type\tpositive_type\tnegative_type\n")
                for info in last_triplet_info:
                    f.write(f"{info[0]}\t{info[1]}\t{info[2]}\n")
            print(f"Saved triplet info for seed {seed}")
    
    # Save summary of all seeds
    all_objectives_file = base_results_dir + "all_objectives.txt"
    with open(all_objectives_file, 'w') as f:
        for seed, obj_value in enumerate(all_seeds_objectives):
            f.write(f"Seed {seed}: {obj_value}\n")
        f.write(f"\nAverage: {np.mean(all_seeds_objectives)}\n")
        f.write(f"Std: {np.std(all_seeds_objectives)}\n")


if __name__ == "__main__":
    main()