import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
species_map = {i: species for i, species in enumerate(iris.target_names)}
df['species'] = df['species'].map(species_map)

# Display first few rows
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())

# No missing values in the Iris dataset, but if there were:
# df = df.dropna()  # Drop missing values
# df.fillna(df.mean(), inplace=True)  # Fill missing values with mean

# Task 2: Basic Data Analysis
# Compute basic statistics
print(df.describe())

# Grouping by species and computing mean of numerical columns
species_mean = df.groupby('species').mean()
print(species_mean)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# Line Chart (Simulating a trend by adding an index)
df['index'] = range(len(df))
sns.lineplot(data=df, x='index', y='sepal length (cm)', hue='species')
plt.title('Sepal Length Trend Over Time')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend(title='Species')
plt.show()

# Bar Chart
plt.figure(figsize=(8, 6))
sns.barplot(x=species_mean.index, y=species_mean['sepal length (cm)'])
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal length (cm)'], bins=15, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
