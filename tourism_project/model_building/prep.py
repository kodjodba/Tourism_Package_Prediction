# ------------------------------------------------------------------
# Data preparation: clean the registered dataset and create stratified
# train/test splits. The CSVs are uploaded as a workflow artifact.
# ------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"   # load directly from the repo data folder
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 1) Remove unnecessary columns: saved pandas index + customer identifier (not predictive)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# 2) Fix inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 3) Remove duplicate records
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_dup} duplicate rows")

# 4) Impute any missing values (defensive - keeps the pipeline robust to future data)
for col in df.columns:
    if df[col].isna().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "ProdTaken"  # 1 if the customer purchased the package, else 0
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Clean shape:", df.shape)
print(f"Data prepared: train {Xtrain.shape}, test {Xtest.shape} - splits written.")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
