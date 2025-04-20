import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model_year", "origin", "car_name"]

df = pd.read_csv("data/auto-mpg_data-original.csv", names=columns, sep=r'\s+', na_values='?')

# Convert horsepower to numeric and handle missing values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# Drop rows with missing target values before modeling
df = df.dropna(subset=['mpg'])

print("Column names:", df.columns.tolist())


# Clean the data
df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'])
df.dropna(subset=['horsepower'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Add new feature
df['power_to_weight'] = df['horsepower'] / df['weight']

# ----- Visualization Section -----

# 1. Histograms
df.hist(bins=30, figsize=(12, 8))
plt.suptitle("Histograms of Auto MPG Data", fontsize=16)
plt.show()

# 2. Boxenplots
for column in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(6, 4))
    sns.boxenplot(data=df[column])
    plt.title(f'Boxenplot for {column}')
    plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop(columns=['car_name']).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# 3. Pairplot (scatterplot matrix)
sns.pairplot(df)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# 4. Example scatterplot: Horsepower vs MPG by Origin
plt.scatter(df['horsepower'], df['mpg'], c=df['origin'], cmap='viridis', alpha=0.6)
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Horsepower vs MPG by Origin')
plt.colorbar(label='Origin')
plt.show()

# 5. Histogram of MPG
sns.histplot(df['mpg'], kde=True, bins=30)
plt.title('MPG Distribution')
plt.xlabel('MPG')
plt.show()

# 6. Countplot for cylinders
sns.countplot(x='cylinders', data=df, palette='pastel')
plt.title('Count of Cars by Cylinder Count')
plt.xlabel('Cylinders')
plt.ylabel('Count')
plt.show()

# ----- Regression Section -----

# Feature sets
X1 = df[['horsepower']]
X2 = df[['weight']]
X3 = df[['horsepower', 'weight']]
X4 = df[['acceleration']]
y = df['mpg']

# Train-test split
X1_train, X1_test, y_train1, y_test1 = train_test_split(X1, y, test_size=0.2, random_state=123)
X2_train, X2_test, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=123)
X3_train, X3_test, y_train3, y_test3 = train_test_split(X3, y, test_size=0.2, random_state=123)
X4_train, X4_test, y_train4, y_test4 = train_test_split(X4, y, test_size=0.2, random_state=123)

# Train models
lr1 = LinearRegression().fit(X1_train, y_train1)
lr2 = LinearRegression().fit(X2_train, y_train2)
lr3 = LinearRegression().fit(X3_train, y_train3)
lr4 = LinearRegression().fit(X4_train, y_train4)

# Predictions
y_pred1 = lr1.predict(X1_test)
y_pred2 = lr2.predict(X2_test)
y_pred3 = lr3.predict(X3_test)
y_pred4 = lr4.predict(X4_test)

# Evaluation function
def evaluate_model(y_true, y_pred, label):
    print(f"\n{label}")
    print("R² Score:", r2_score(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred) ** 0.5)
    print("MAE:", mean_absolute_error(y_true, y_pred))

# Evaluate each case
evaluate_model(y_test1, y_pred1, "Case 1: Horsepower")
evaluate_model(y_test2, y_pred2, "Case 2: Weight")
evaluate_model(y_test3, y_pred3, "Case 3: Horsepower + Weight")
evaluate_model(y_test4, y_pred4, "Case 4: Acceleration")

# Visualize regression results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test1, y_pred1, alpha=0.6)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Case 1: Horsepower")

plt.subplot(1, 2, 2)
plt.scatter(y_test3, y_pred3, alpha=0.6, color='orange')
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Case 3: Horsepower + Weight")

plt.tight_layout()
plt.show()
