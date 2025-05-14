# import required libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_california_housing 

def main():
    try:
        # Load the California Housing dataset 
        california_housing = fetch_california_housing() 
        data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names) 
        
        # Compute the correlation matrix 
        corr_matrix = data.corr() 
        print("\nCorrelation Matrix:")
        print(corr_matrix) 
        
        # Visualize the correlation matrix using a heatmap 
        plt.figure(figsize=(8, 6)) 
        sns.heatmap(corr_matrix, annot=True, cmap='crest', fmt='.2f') 
        plt.title('Correlation Matrix Heatmap') 
        plt.tight_layout()
        plt.show() 
        plt.close()
        
        # Create a pair plot to visualize pairwise relationships between features 
        plt.figure() 
        sns.pairplot(data, kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.5}) 
        plt.show() 
        plt.close()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 

