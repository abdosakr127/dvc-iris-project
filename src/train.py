import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load train data
train = pd.read_csv("data/train.csv")

X_train = train.drop("species", axis=1)
y_train = train["species"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")