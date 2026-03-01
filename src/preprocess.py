import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw data
df = pd.read_csv("data/raw.csv")

# Split features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save train and test
train = X_train.copy()
train["species"] = y_train
train.to_csv("data/train.csv", index=False)

test = X_test.copy()
test["species"] = y_test
test.to_csv("data/test.csv", index=False)