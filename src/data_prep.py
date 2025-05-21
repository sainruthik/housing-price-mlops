import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/ames.csv"):
    df = pd.read_csv(path)
    
    # Drop columns with too many nulls or IDs
    cols_to_drop = ["Alley", "PoolQC", "Fence", "MiscFeature", "Id"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])


    # Fill NA
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categoricals simply (label encode for now)
    df = pd.get_dummies(df, drop_first=True)

    # Split
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
