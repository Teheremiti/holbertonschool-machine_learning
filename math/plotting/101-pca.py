#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

pca_data_T = pca_data.T

ax = plt.axes(projection='3d')
ax.set_title('PCA of Iris Dataset')

ax.scatter(pca_data_T[0], pca_data_T[1],
           pca_data_T[2], c=labels, cmap=plt.cm.plasma)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

plt.tight_layout()
plt.show()
