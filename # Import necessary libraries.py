# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (use your own file path)
file_path = 'C:/Users/user/Desktop/Survival analysis/kidney_disease.csv'  # Adjust based on your file path
kidney_data = pd.read_csv(file_path)

# Replace '?' with NaN to handle missing values
kidney_data.replace('?', np.nan, inplace=True)

# Display the first few rows of the dataset
print(kidney_data.head())

# Check for missing values
print("\nMissing values in each column:")
print(kidney_data.isnull().sum())

# Statistical summary of numerical columns
print("\nStatistical summary of numerical columns:")
print(kidney_data.describe())

# Statistical summary of categorical columns
print("\nStatistical summary of categorical columns:")
print(kidney_data.describe(include=['O']))

# Handle missing values (simple method: drop rows with missing values)
kidney_data_clean = kidney_data.dropna()

# Check the distribution of patients with and without CKD (assuming 'classification' column represents CKD status)
print("\nCKD Status Distribution:")
print(kidney_data_clean['classification'].value_counts())

# Visualize CKD vs Non-CKD patients
plt.figure(figsize=(6, 4))
sns.countplot(data=kidney_data_clean, x='classification')
plt.title('Distribution of CKD vs Non-CKD Patients')
plt.show()

# Visualize the distribution of numerical features
numerical_columns = kidney_data_clean.select_dtypes(include=[np.number]).columns.tolist()

plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sns.histplot(kidney_data_clean[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Correlation Heatmap: Relationships between numerical variables
plt.figure(figsize=(10, 8))
sns.heatmap(kidney_data_clean[numerical_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

# Boxplots: Distribution of numerical features grouped by CKD status
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(data=kidney_data_clean, x='classification', y=col)
    plt.title(f'{col} by CKD Status')
plt.tight_layout()
plt.show()

# Pairplot for relationships between variables (selected variables)
sns.pairplot(kidney_data_clean, vars=['age', 'bp', 'bgr', 'hemo'], hue='classification', diag_kind='kde')
plt.title('Pairplot of Selected Features')
plt.show()

# Violin plots: Visualize distribution and variance of features by CKD status
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns[:6], 1):  # Taking the first 6 columns for readability
    plt.subplot(2, 3, i)
    sns.violinplot(data=kidney_data_clean, x='classification', y=col)
    plt.title(f'Violin plot of {col} by CKD Status')
plt.tight_layout()
plt.show()
