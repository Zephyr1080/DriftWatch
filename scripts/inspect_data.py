import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data/raw/hour.csv")

# Check whether the file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

# 1. Basic information
print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 2. First five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 3. Column names and data types
print("\n========== COLUMN INFORMATION ==========")
print(df.info())

# 4. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 5. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# 6. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe().T)

# 7. Target variable distribution
print("\n========== TARGET DISTRIBUTION ==========")
print(df["cnt"].describe())

# 8. Date range
print("\n========== DATE RANGE ==========")
df["datetime"] = pd.to_datetime(df["dteday"])
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())

# 9. Unique values in categorical features
print("\n========== CATEGORICAL FEATURE DISTRIBUTION ==========")
for column in ["season", "weathersit", "holiday", "workingday"]:
    print(f"\n{column}:")
    print(df[column].value_counts().sort_index())

print("\n========== INSPECTION COMPLETE ==========")