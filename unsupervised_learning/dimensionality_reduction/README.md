# Dimensionality Reduction

This project implements various dimensionality reduction techniques used in machine learning and data analysis. Dimensionality reduction is crucial for dealing with high-dimensional data by reducing computational complexity, mitigating the curse of dimensionality, and enabling data visualization.

## Overview

Dimensionality reduction techniques can be broadly categorized into:

### Linear Methods
- **Principal Component Analysis (PCA)**: Projects data onto orthogonal axes that maximize variance
- **Linear Discriminant Analysis (LDA)**: Finds projections that maximize class separability

### Non-linear Methods
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Preserves local neighborhood structure for visualization
- **Kernel PCA**: Non-linear extension of PCA using kernel methods

## Project Tasks

| File | Task | Description |
|------|------|-------------|
| **0-pca.py** | **PCA with Variance Preservation** | Performs PCA while maintaining a specified fraction of the original variance. Returns the transformation matrix that achieves the desired variance retention. |
| **1-pca.py** | **PCA with Fixed Dimensions** | Performs PCA and transforms data to a specified number of dimensions. Returns the transformed dataset projected onto the principal components. |

## Key Concepts

### Principal Component Analysis (PCA)
PCA finds orthogonal axes (principal components) that capture maximum variance in the data. The first principal component explains the most variance, the second explains the most remaining variance, and so on.

**Mathematical Foundation:**
- Uses Singular Value Decomposition (SVD) for numerical stability
- Eigenvalues of the covariance matrix represent explained variance
- Eigenvectors define the principal component directions

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE is particularly effective for visualizing high-dimensional data by preserving local neighborhood structures while revealing global patterns.

**Key Features:**
- Converts similarities to joint probabilities in high dimensions
- Uses Student's t-distribution in low dimensions to avoid crowding
- Optimizes Kullback-Leibler divergence between probability distributions

## Implementation Notes

All implementations prioritize:
- Numerical stability using SVD over eigendecomposition
- Computational efficiency with vectorized operations
- Memory optimization for large datasets
- Proper documentation and type hints
