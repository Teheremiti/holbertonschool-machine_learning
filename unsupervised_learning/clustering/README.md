# Clustering - Unsupervised Learning

![Clustering Banner]()

## 📖 Overview

Clustering is a fundamental unsupervised machine learning technique that groups data points into clusters based on their similarities. Unlike supervised learning, clustering doesn't require labeled data - instead, it discovers hidden patterns and structures within datasets by organizing similar data points together.

This project implements various clustering algorithms from scratch, providing deep insights into their mathematical foundations and practical applications.

## 🎯 Learning Objectives

By the end of this project, you will understand:

- **K-means Clustering**: Centroid-based partitioning algorithm
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering with EM algorithm
- **Expectation-Maximization (EM)**: Iterative optimization for parameter estimation
- **Model Selection**: Using BIC for optimal cluster number determination
- **Hierarchical Clustering**: Agglomerative clustering with dendrograms
- **Scikit-learn Integration**: Leveraging industry-standard implementations

## 🛠️ Requirements

### General
- **Operating System**: Ubuntu 20.04 LTS
- **Python Version**: 3.9
- **Code Style**: pycodestyle (version 2.11.1)
- **File Requirements**:
  - All files end with a new line
  - First line: `#!/usr/bin/env python3`
  - All files must be executable

### Dependencies
```bash
numpy==1.25.2
sklearn==1.5.0
scipy==1.11.4
matplotlib==3.5.0
```

### Imports
- **Only allowed imports**: `import numpy as np` (unless specified otherwise)
- **No loops allowed** in most functions (where specified)
- **Documentation required** for all modules, classes, and functions

## 🧠 Clustering Algorithms Implemented

### 1. **K-means Clustering**
A centroid-based algorithm that partitions data into k clusters by minimizing within-cluster sum of squares.

**Key Features:**
- Initialization using uniform distribution
- Iterative centroid updates
- Convergence detection
- Variance calculation for cluster quality assessment

### 2. **Gaussian Mixture Models (GMM)**
A probabilistic model that assumes data comes from a mixture of Gaussian distributions.

**Key Features:**
- Expectation-Maximization (EM) algorithm
- Probability density function calculation
- Maximum likelihood parameter estimation
- Soft clustering with posterior probabilities

### 3. **Hierarchical Clustering**
An agglomerative approach that builds a tree of clusters using Ward linkage.

**Key Features:**
- Ward linkage method
- Dendrogram visualization
- Distance-based cluster formation
- No need to specify number of clusters in advance

## 📁 Project Structure

```
clustering/
├── README.md                     # This comprehensive guide
├── 0-initialize.py              # K-means centroid initialization
├── 1-kmeans.py                  # K-means clustering algorithm
├── 2-variance.py                # Intra-cluster variance calculation
├── 3-optimum.py                 # Optimal k selection by variance
├── 4-initialize.py              # GMM parameter initialization
├── 5-pdf.py                     # Gaussian PDF calculation
├── 6-expectation.py             # EM expectation step
├── 7-maximization.py            # EM maximization step
├── 8-EM.py                      # Complete EM algorithm
├── 9-BIC.py                     # BIC-based model selection
├── 10-kmeans.py                 # Sklearn K-means wrapper
├── 11-gmm.py                    # Sklearn GMM wrapper
└── 12-agglomerative.py          # Scipy agglomerative clustering
```

## 📋 Tasks Overview

| # | Task | Algorithm | Description | Key Concepts |
|---|------|-----------|-------------|--------------|
| **0** | **Initialize K-means** | K-means | Initialize cluster centroids using multivariate uniform distribution | Uniform sampling, centroid initialization |
| **1** | **K-means Algorithm** | K-means | Complete K-means clustering with convergence detection | Iterative optimization, centroid updates |
| **2** | **Variance Calculation** | K-means | Calculate total intra-cluster variance for model evaluation | Distance metrics, cluster quality |
| **3** | **Optimal K Selection** | K-means | Find optimal number of clusters using variance analysis | Elbow method, model comparison |
| **4** | **GMM Initialization** | GMM | Initialize priors, means, and covariances for GMM | Parameter initialization, K-means seeding |
| **5** | **Gaussian PDF** | GMM | Calculate multivariate Gaussian probability density function | Probability theory, matrix operations |
| **6** | **Expectation Step** | EM | Compute posterior probabilities using Bayes' theorem | Bayes' theorem, responsibility calculation |
| **7** | **Maximization Step** | EM | Update GMM parameters using weighted maximum likelihood | MLE, parameter updates |
| **8** | **Complete EM Algorithm** | EM | Full expectation-maximization with convergence monitoring | Iterative optimization, convergence criteria |
| **9** | **BIC Model Selection** | GMM | Find optimal clusters using Bayesian Information Criterion | Model selection, complexity penalty |
| **10** | **Sklearn K-means** | K-means | Professional K-means using scikit-learn | Industry standards, API usage |
| **11** | **Sklearn GMM** | GMM | Professional GMM using scikit-learn | Advanced implementations, BIC scoring |
| **12** | **Agglomerative Clustering** | Hierarchical | Hierarchical clustering with dendrogram visualization | Ward linkage, dendrogram analysis |

## 🚀 Usage Examples

### K-means Clustering
```python
#!/usr/bin/env python3
import numpy as np
from kmeans import kmeans
from initialize import initialize

# Generate sample data
np.random.seed(0)
X = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=100)

# Initialize centroids
centroids = initialize(X, k=3)

# Perform K-means clustering
final_centroids, cluster_assignments = kmeans(X, k=3)
print(f"Centroids shape: {final_centroids.shape}")
print(f"Cluster assignments: {cluster_assignments}")
```

### Gaussian Mixture Model
```python
#!/usr/bin/env python3
import numpy as np
from expectation_maximization import expectation_maximization

# Generate sample data
np.random.seed(11)
X = np.random.multivariate_normal([20, 30], [[50, 10], [10, 50]], size=1000)

# Fit GMM using EM algorithm
pi, m, S, g, log_likelihood = expectation_maximization(X, k=2, verbose=True)
print(f"Final log likelihood: {log_likelihood}")
print(f"Cluster priors: {pi}")
```

### BIC Model Selection
```python
#!/usr/bin/env python3
import numpy as np
from BIC import BIC

# Generate multi-cluster data
np.random.seed(42)
X = np.concatenate([
    np.random.normal([0, 0], 1, (100, 2)),
    np.random.normal([5, 5], 1, (100, 2)),
    np.random.normal([10, 0], 1, (100, 2))
])

# Find optimal number of clusters
best_k, best_result, likelihoods, bic_values = BIC(X, kmin=1, kmax=10)
print(f"Optimal number of clusters: {best_k}")
print(f"BIC values: {bic_values}")
```

## 📊 Mathematical Foundations

### K-means Objective Function
Minimize within-cluster sum of squares:
```
J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²
```

### Gaussian Mixture Model
Probability density function:
```
p(x) = Σⱼ₌₁ᵏ πⱼ 𝒩(x | μⱼ, Σⱼ)
```

### EM Algorithm Updates
**E-step:** γᵢⱼ = (πⱼ 𝒩(xᵢ | μⱼ, Σⱼ)) / (Σₖ πₖ 𝒩(xᵢ | μₖ, Σₖ))

**M-step:**
- πⱼ = (1/n) Σᵢ γᵢⱼ
- μⱼ = (Σᵢ γᵢⱼ xᵢ) / (Σᵢ γᵢⱼ)
- Σⱼ = (Σᵢ γᵢⱼ (xᵢ - μⱼ)(xᵢ - μⱼ)ᵀ) / (Σᵢ γᵢⱼ)

### BIC Formula
```
BIC = p × ln(n) - 2 × ℓ
```
Where p = number of parameters, n = sample size, ℓ = log-likelihood

## 🎨 Visualization Examples

The project includes visualization capabilities for:
- **Scatter plots** with cluster coloring
- **Centroids** marked with special symbols
- **Dendrograms** for hierarchical clustering
- **BIC curves** for model selection

## ⚡ Performance Considerations

### Optimization Techniques Used:
- **Vectorized operations** with NumPy for efficiency
- **Broadcasting** for matrix operations
- **Early convergence detection** in iterative algorithms
- **Minimal loops** as per project requirements
- **Memory-efficient** implementations

### Complexity Analysis:
- **K-means**: O(nkdi) where n=samples, k=clusters, d=dimensions, i=iterations
- **GMM/EM**: O(nkd²i) due to covariance matrix operations
- **Hierarchical**: O(n³) for complete linkage matrix construction

## 🧪 Testing and Validation

Each implementation includes:
- **Input validation** for robustness
- **Edge case handling** for stability
- **Mathematical correctness** verification
- **Expected output validation** against examples
- **Performance benchmarking** where applicable

## 🔗 Dependencies and Integration

### From Scratch Implementations:
- Pure NumPy mathematical implementations
- Custom convergence criteria
- Manual parameter optimization

### Scikit-learn Integration:
- Professional-grade implementations
- Optimized performance
- Standardized APIs
- Advanced features (BIC scoring, etc.)

## 🎓 Educational Value

This project demonstrates:
- **Mathematical rigor** in algorithm implementation
- **Software engineering** best practices
- **Documentation standards** for maintainability
- **Performance optimization** techniques
- **Industry integration** with established libraries

## 📚 References and Further Reading

- **K-means**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations
- **EM Algorithm**: Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm
- **BIC**: Schwarz, G. (1978). Estimating the dimension of a model
- **Ward Linkage**: Ward Jr, J.H. (1963). Hierarchical grouping to optimize an objective function

---

**Course**: Holberton School - Machine Learning Specialization
**Project**: Unsupervised Learning - Clustering
