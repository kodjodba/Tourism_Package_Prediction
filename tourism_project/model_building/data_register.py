# ------------------------------------------------------------------
# Data registration: validate the raw dataset stored in the repo and
# print a summary. Exits with an error if the schema is not as expected,
# which stops the GitHub Actions pipeline before any training happens.
# ------------------------------------------------------------------
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # raw dataset inside the repo data folder

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# The target must be binary (0/1)
if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must only contain 0 and 1")

extra = [c for c in df.columns if c not in expected_columns]

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Extra (non-dictionary) columns:", extra if extra else "None")
print("Total missing values:", int(df.isna().sum().sum()))
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())
