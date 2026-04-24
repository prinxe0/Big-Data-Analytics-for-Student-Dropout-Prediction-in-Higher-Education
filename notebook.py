import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# LOAD DATASET

df = pd.read_csv("student_mental_health_burnout_1M.csv")

print("Dataset Shape:", df.shape)
print("\nPreview:")
print(df.head())

# DATA EXPLORATION (EDA)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Info:")
print(df.info())

# DATA CLEANING

# Remove duplicates
df = df.drop_duplicates()

df.fillna(df.mean(numeric_only=True), inplace=True)

df.fillna(method='ffill', inplace=True)

print("\nAfter Cleaning Shape:", df.shape)


df = pd.get_dummies(df, columns=['gender', 'risk_level'])

print("\nColumns after encoding:")
print(df.columns)


numerical_df = df.select_dtypes(include=[np.number])

model = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = model.fit_predict(numerical_df)

print("\nOutlier Distribution:")
print(df['outlier'].value_counts())


plt.figure()
df['dropout_risk'].hist()
plt.title("Dropout Risk Distribution")
plt.xlabel("Dropout Risk")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.scatter(df['stress_level'], df['dropout_risk'])
plt.title("Stress Level vs Dropout Risk")
plt.xlabel("Stress Level")
plt.ylabel("Dropout Risk")
plt.show()

plt.figure()
plt.scatter(df['sleep_hours'], df['dropout_risk'])
plt.title("Sleep Hours vs Dropout Risk")
plt.xlabel("Sleep Hours")
plt.ylabel("Dropout Risk")
plt.show()

plt.figure()
plt.scatter(df['burnout_score'], df['dropout_risk'])
plt.title("Burnout Score vs Dropout Risk")
plt.xlabel("Burnout Score")
plt.ylabel("Dropout Risk")
plt.show()

plt.figure()
df['outlier'].value_counts().plot(kind='bar')
plt.title("Outlier Distribution")
plt.xlabel("Outlier Class (-1 = Outlier, 1 = Normal)")
plt.ylabel("Count")
plt.show()

# Recompute numerical_df AFTER adding outlier
numerical_df = df.select_dtypes(include=[np.number])
corr = numerical_df.corr()

plt.figure()
plt.imshow(corr)
plt.title("Correlation Heatmap")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

output_path = r"C:\Users\Prince\Documents\Big Data\Deliverables\archive\cleaned_student_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved at:")
print(output_path)

# Verify file exists
print("File Exists:", os.path.exists(output_path))