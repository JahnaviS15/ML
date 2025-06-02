# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()
X = iris.data # Features
y = iris.target # Target labels
label_names = iris.target_names
# Create a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
# Apply PCA to reduce dimensionality from 4 to 2
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=principal_components,
columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y
# Visualize the result
plt.figure(figsize=(6, 4))
colors = ['y', 'b', 'g']
for i, label in enumerate(np.unique(y)):
plt.scatter(df_pca[df_pca['Target'] == label]['Principal Component 1'],
df_pca[df_pca['Target'] == label]['Principal Component 2'],
label=label_names[label],
color=colors[i])
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('pcaofirisdataset.png')
plt.show()
