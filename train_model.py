from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("../data/leads.csv")

X = df.drop(columns=["Intent", "Email", "Phone_Number"])
y = df["Intent"]

numeric_features = ["Credit_Score", "Income"]
categorical_features = ["Age_Group", "Family_Background"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=100))
])

model.fit(X, y)

# Save model
joblib.dump(model, "../model/gbc_lead_scorer.pkl")
print("Model saved at model/gbc_lead_scorer.pkl")