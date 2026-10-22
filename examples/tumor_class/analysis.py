import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/tumor_class/data.csv"

features = [f"feature_{i}" for i in range(1, 31)]
df = pd.read_csv(DATA_PATH, header=None, names=["id", "diagnosis"] + features)
 
print("Размер:", df.shape)
print("Пропуски:", df.isna().sum().sum())
print("Дубликаты:", df.duplicated(subset=features).sum(), end="\n\n")
 
print("Баланс классов:")
print(df["diagnosis"].value_counts(normalize=True).round(3), end="\n\n")
 
print("Статистика признаков:")
print(df[features].agg(["min", "max", "std"]).T)
 
corr = df[features].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(30), range(1, 31))
plt.yticks(range(30), range(1, 31))
plt.colorbar()
plt.title("Корреляция признаков")
plt.show()
