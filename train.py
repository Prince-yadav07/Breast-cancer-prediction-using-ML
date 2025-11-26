import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Select EXACT same 5 features used in Flask
features = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean compactness",
    "mean concavity"
]

X = df[features]

# Scale inputs (important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save BOTH model + scaler
pickle.dump({"model": clf, "scaler": scaler}, open("model.pkl", "wb"))

print("Model trained & saved successfully!")
