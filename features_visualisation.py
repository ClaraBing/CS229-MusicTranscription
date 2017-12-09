import numpy as np

# Sciki-learn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
features_file = 'train_features_mtrx.npy'
annotations_file = 'train_features_annotations.npy'
M = 50 # Intermediate dimension of the data

# Load data
features = np.load(features_file)
annotations = np.load(annotations_file)
N = len(features_file)
print (N)
print (features.shape[1])

# Apply PCA to each of the features
pca = PCA(n_components=M)
pca.fit(features)
transformed_features = pca.transform(features)
print (transformed_features.shape)


# Apply t-SNE to the transformed features for visualisation
embedded_features = TSNE(n_components=3).fit_transform(transformed_features)
print (embedded_features.shape)

# Visualise data

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = plt.subplot(111, projection='3d')

for i in range(109):
    if i in annotations:
        batch = embedded_features[annotations==i]
        ax.plot(batch[:,0], batch[:,1], batch[:,2], 'o', label=str(i))
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
plt.show()
