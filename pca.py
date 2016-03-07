import numpy as np
from sklearn.decomposition import PCA

def get_pca(data_sets, n_component):
	pca = PCA(n_components=n_component)
    pca.fit(X)
    return pca.explained_variance_ratio_

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca = PCA(n_components=2)
# pca.fit(X)
# print(pca.explained_variance_ratio_)