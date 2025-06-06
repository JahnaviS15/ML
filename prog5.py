import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter 
data = np.random.rand(100) 
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]] 
def euclidean_distance(x1, x2): 
 return abs(x1 - x2) 
def knn_classifier(train_data, train_labels, test_point, k): 
 distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) 
 for i in range(len(train_data))] 
 distances.sort(key=lambda x: x[0]) 
 k_nearest_neighbors = distances[:k] 
 k_nearest_labels = [label for _, label in k_nearest_neighbors] 
 return Counter(k_nearest_labels).most_common(1)[0][0] 
train_data = data[:50] 
train_labels = labels 
test_data = data[50:] 
k_values = [1, 2, 3, 4, 5, 20, 30] 
print("--- k-Nearest Neighbors Classification ---") 
print("Training dataset: First 50 points labeled based on the rule (x <= 0.5 -> Class1, x > 0.5 -> Class2)")
print("Testing dataset: Remaining 50 points to be classified\n")
results = {} 
for k in k_values: 
    print(f"Results for k = {k}:") 
    classified_labels = [knn_classifier(train_data, train_labels, test_point, k) for test_point in test_data] 
    results[k] = classified_labels 
    for i, label in enumerate(classified_labels, start=51): 
        print(f"Point x{i} (value: {test_data[i - 51]:.4f}) is classified as {label}") 
    print("\n") 
print("Classification complete.\n") 
# VISUALIZING THE CLASSIFICATION: 
num_k = len(k_values) 
rows = (num_k + 2) // 3 # Dynamically calculate the number of rows for the grid 
cols = 3 # Fixed number of columns for a 3xN grid 
plt.figure(figsize=(15, 5 * rows)) # Adjust figure size based on rows 
for idx, k in enumerate(k_values): 
 classified_labels = results[k] 
 class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class1"] 
 class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class2"] 
 # Create a subplot for each value of k 
 plt.subplot(rows, cols, idx + 1) 
 plt.scatter(train_data, [0] * len(train_data), c=["blue" if label == "Class1" else "red" for label in train_labels], 
 label="Training Data", marker="o") 
 plt.scatter(class1_points, [1] * len(class1_points), c="blue", label="Class1 (Test)", marker="x") 
 plt.scatter(class2_points, [1] * len(class2_points), c="red", label="Class2 (Test)", marker="x") 
 plt.title(f"k-NN Results for k = {k}") 
 plt.xlabel("Data Points") 
 plt.ylabel("Classification Level") 
 plt.legend() 
 plt.grid(True) 
# Adjust layout for better spacing 
plt.tight_layout() 
plt.savefig('knn_classification.png')
plt.show()