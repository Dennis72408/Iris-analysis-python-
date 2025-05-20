iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# Explore the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nCheck for missing values:")
print(df.isnull().sum())

# Basic Analysis
species_counts = df['species'].value_counts()
print("\nSpecies counts:")
print(species_counts)

# Visualizations
sns.set(style="whitegrid")

# 1. Histogram of each feature
df.hist(figsize=(10, 8), edgecolor='black')
plt.suptitle("Histograms of Iris Features", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# 2. Pairplot colored by species
sns.pairplot(df, hue="species", diag_kind="kde")
plt.suptitle("Pairplot of Iris Features", fontsize=16)
plt.show()

# 3. Boxplot for each feature by species
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f"{feature} by Species")
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

