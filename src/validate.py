import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
test = pd.read_csv("data/test.csv")

X_test = test.drop("species", axis=1)
y_test = test["species"]

# Load model
model = joblib.load("models/model.pkl")

# Predict
preds = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, preds)

# Save metrics
metrics = {"accuracy": accuracy}

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(y_test, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("plots/confusion_matrix.png")