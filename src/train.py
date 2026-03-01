import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load train data
train = pd.read_csv("data/train.csv")

X_train = train.drop("species", axis=1)
y_train = train["species"]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")