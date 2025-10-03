
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# 1. Load Dataset
# ------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Convert to pandas DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("First 5 rows of dataset:")
print(df.head())

# ------------------------------
# 2. Data Preprocessing
# ------------------------------
# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for many ML models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 3. Model Training
# ------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ------------------------------
# 4. Model Evaluation
# ------------------------------
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 5. Visualization
# ------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
