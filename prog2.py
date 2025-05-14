# import required libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 
# Load the California Housing dataset 
california_housing = fetch_california_housing() 
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names) 
# # Compute the correlation matrix 
corr_matrix = data.corr() 
print(corr_matrix) 
# # Note: default method to calculate correlation is Pearson's correlation coefficient. 
# # Visualize the correlation matrix using a heatmap 
plt.figure(figsize=(8, 6)) 
sns.heatmap(corr_matrix, annot=True, cmap= 'crest', fmt='.2f') 
plt.title('Correlation Matrix Heatmap') 
plt.show() 
# # Create a pair plot to visualize pairwise relationships between features 
plt.figure() 
sns.pairplot(data, kind='scatter',diag_kind='kde', plot_kws={'alpha': 0.5}) 
plt.show() 

