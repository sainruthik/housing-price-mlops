import pandas as pd
from src.data_prep import load_and_preprocess

# Load processed dataset (only to get column structure)
_, X_test, _, _ = load_and_preprocess()

# Create a zero-filled row with all expected features
template = pd.DataFrame(columns=pd.DataFrame(X_test).columns)
template.loc[0] = 0

# Save template
template.to_csv("src/input_template.csv", index=False)
print("Template saved with shape:", template.shape)
