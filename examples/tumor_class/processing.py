import numpy as np
import pandas as pd

DATA_PATH = "data/tumor_class/data.csv"
TRAIN_PATH = "data/tumor_class/train.csv"
VALIDATION_PATH = "data/tumor_class/validation.csv"

features = [f"feature_{i}" for i in range(1, 31)]
df = pd.read_csv(DATA_PATH, header=None, names=["id", "diagnosis"] + features)

rng = np.random.default_rng(42)
indices = rng.permutation(len(df))
split = int(0.7 * len(df))

train_data = df.iloc[indices[:split]]
validation_data = df.iloc[indices[split:]]
print("train:", len(train_data), "valid:", len(validation_data))
print("Доля M в train:", round((train_data["diagnosis"] == "M").mean(), 3))
print("Доля M в valid:", round((validation_data["diagnosis"] == "M").mean(), 3))

train_data.to_csv(TRAIN_PATH, index=False)
validation_data.to_csv(VALIDATION_PATH, index=False)
