import numpy as np
from numpy import linalg as LA
# Sciki-learn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# Visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
features_file = 'train_features_mtrx.npy'
annotations_file = 'train_features_annotations.npy'
thres = 0.95
M = 50
# Load data
features = np.load(features_file)
annotations = np.load(annotations_file)
N = len(features_file)
# Standardize features
scaler = StandardScaler()
scaler.fit(features)
print(scaler.mean_.shape)
std_features = scaler.transform(features)

# Apply PCA to each of the features
pca = PCA(n_components=M)
pca.fit(std_features)
transformed_features = pca.transform(std_features)
print (transformed_features.shape)

# Apply t-SNE to the transformed features for visualisation
embedded_features = TSNE(n_components=2).fit_transform(transformed_features)
print (embedded_features.shape)

# Visualise data

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
L = len(set(annotations))
selected_bins = [0]
selected_bins.extend(range(39, 47))
selected_bins.extend(range(58, 69))
selected_bins.extend(range(75, 80))

# Generate L random colors
colors = [(np.random.randint(0,255)/255, np.random.randint(0,255)/255, np.random.randint(0,255)/255) for i in range(len(selected_bins))]
for idx, i in enumerate(selected_bins):
    batch = embedded_features[annotations==i]
    if len(batch) < 10:
        continue
    plt.scatter(batch[:,0], batch[:,1], label=str(i), c=colors[idx])
plt.legend(loc='upper left', numpoints=1, ncol=1, fontsize=12)
plt.show()
