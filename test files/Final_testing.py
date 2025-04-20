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

#df = pd.read_csv("data/auto-mpg_data-original.csv", names=columns, sep=r'\s+', na_values='?')
df = pd.read_csv("../data/auto-mpg_data-original.csv", names=columns, sep=r'\s+', na_values='?')

# Display the first 10 rows
print(df.head(10))

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

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
sns.countplot(x='cylinders', hue='cylinders', data=df, palette='pastel', legend=False)
plt.title('Count of Cars by Cylinder Count')
plt.xlabel('Cylinders')
plt.ylabel('Count')
plt.show()

# Convert horsepower to numeric and handle missing values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Drop rows with missing target values before modeling
df = df.dropna(subset=['mpg'])
print("Column names:", df.columns.tolist())

# Clean the data
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Add new feature
df['power_to_weight'] = df['horsepower'] / df['weight']

# ----- Regression Section -----

# Feature sets (after splitting data into train/test)
X1 = df[['horsepower']]  # Input feature
X2 = df[['weight']]  # Input feature
X3 = df[['horsepower', 'weight']]  # Input features
X4 = df[['acceleration']]  # Input feature
y = df['mpg']  # Target variable

# Train-test split (before scaling)
X1_train, X1_test, y_train1, y_test1 = train_test_split(X1, y, test_size=0.2, random_state=123)
X2_train, X2_test, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=123)
X3_train, X3_test, y_train3, y_test3 = train_test_split(X3, y, test_size=0.2, random_state=123)
X4_train, X4_test, y_train4, y_test4 = train_test_split(X4, y, test_size=0.2, random_state=123)

# Apply StandardScaler to the feature sets (scaling after split)
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

scaler3 = StandardScaler()
X3_train_scaled = scaler3.fit_transform(X3_train)
X3_test_scaled = scaler3.transform(X3_test)

scaler4 = StandardScaler()
X4_train_scaled = scaler4.fit_transform(X4_train)
X4_test_scaled = scaler4.transform(X4_test)

# Train models using a function to reduce redundancy
def train_and_evaluate(X_train, X_test, y_train, y_test, label):
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation function
    print(f"\n{label}")
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("MAE:", mean_absolute_error(y_test, y_pred))

# Model training and evaluation for all cases
train_and_evaluate(X1_train_scaled, X1_test_scaled, y_train1, y_test1, "Case 1: Horsepower")
train_and_evaluate(X2_train_scaled, X2_test_scaled, y_train2, y_test2, "Case 2: Weight")
train_and_evaluate(X3_train_scaled, X3_test_scaled, y_train3, y_test3, "Case 3: Horsepower + Weight")
train_and_evaluate(X4_train_scaled, X4_test_scaled, y_train4, y_test4, "Case 4: Acceleration")

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

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score

# Define the features
X = df[['horsepower', 'weight', 'acceleration']]  # Use relevant features
y = df['mpg']

# 5.1 Pipeline 1: Imputer → StandardScaler → Linear Regression
pipeline1 = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['horsepower', 'weight', 'acceleration'])
    ])),
    ('model', LinearRegression())
])

# 5.2 Pipeline 2: Imputer → Polynomial Features (degree=3) → StandardScaler → Linear Regression
pipeline2 = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=3)),
            ('scaler', StandardScaler())
        ]), ['horsepower', 'weight', 'acceleration'])
    ])),
    ('model', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 5.3 Compare the performance of both pipelines using cross-validation
def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("MAE:", mean_absolute_error(y_test, y_pred))

# Evaluate both pipelines
print("Evaluating Pipeline 1: Imputer → StandardScaler → Linear Regression")
evaluate_pipeline(pipeline1, X_train, y_train, X_test, y_test)

print("\nEvaluating Pipeline 2: Imputer → Polynomial Features → StandardScaler → Linear Regression")
evaluate_pipeline(pipeline2, X_train, y_train, X_test, y_test)

# Optional: Compare performance using cross-validation for each pipeline
cross_val_score(pipeline1, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_score(pipeline2, X, y, cv=5, scoring='neg_mean_squared_error')

# This will give an idea of how well each model performs on unseen data, including polynomial transformations
