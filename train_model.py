import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("../data/leads.csv")

# Features and target
X = df[[
    "Credit Score",
    "Age Group",
    "Family Background",
    "Income",
    "Lead Source",
    "Product Interest Level",
    "Interaction Frequency"
]]
y = df["Intent"]

# Preprocessor
numeric_features = ["Credit Score", "Income"]
categorical_features = [
    "Age Group",
    "Family Background",
    "Lead Source",
    "Product Interest Level",
    "Interaction Frequency"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=100))
])

# Train
model.fit(X, y)

# Save
joblib.dump(model, "model/gbc_lead_scorer.pkl")
print("✅ Model saved to model/gbc_lead_scorer.pkl")