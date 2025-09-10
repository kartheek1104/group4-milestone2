import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("house price prediction.csv")

# 2. Drop unnecessary columns
df.drop(columns=["Id"], inplace=True)

# Drop columns with too many missing values
df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], inplace=True)

# 3. Handle missing values

# Fill numerical missing values with median
num_imputer = SimpleImputer(strategy="median")
df[["LotFrontage", "MasVnrArea", "GarageYrBlt"]] = num_imputer.fit_transform(
    df[["LotFrontage", "MasVnrArea", "GarageYrBlt"]]
)

# Fill categorical missing values with most frequent
cat_cols = ["MasVnrType","Electrical","BsmtQual","BsmtCond","BsmtExposure",
            "BsmtFinType1","BsmtFinType2","GarageType","GarageFinish",
            "GarageQual","GarageCond"]

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 4. Feature Engineering
df["TotalBsmtSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["TotalBath"] = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]


# 5. Encode Categorical Variables
# Ordinal quality features
qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", 
             "HeatingQC", "KitchenQual", "GarageQual", "GarageCond"]

encoder = LabelEncoder()
for col in qual_cols:
    df[col] = encoder.fit_transform(df[col])

# One-hot encode other categorical features
df = pd.get_dummies(df, drop_first=True)


# 6. Scale Numerical Features
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("SalePrice")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 7. Save cleaned dataset
df.to_csv("cleaned_house_price.csv", index=False)
print("Cleaned dataset saved as cleaned_house_price.csv")

