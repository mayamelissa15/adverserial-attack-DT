import pandas as pd
d3 = pd.read_csv("data/BATADAL_dataset03.csv", skipinitialspace=True)
d4 = pd.read_csv("data/BATADAL_dataset04.csv", skipinitialspace=True)

print("=== dataset03 (train) ===")
print(d3["ATT_FLAG"].value_counts())
print(f"Shape: {d3.shape}")
print(f"NaN: {d3.isnull().sum().sum()}")

print("\n=== dataset04 (test) ===")
print(d4["ATT_FLAG"].value_counts())
print(f"Shape: {d4.shape}")
print(f"NaN: {d4.isnull().sum().sum()}")

print("\n=== Types ===")
print(d3.dtypes)