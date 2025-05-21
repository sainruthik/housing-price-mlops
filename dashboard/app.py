import streamlit as st
import sys
import os

# Add root path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 Housing Price Prediction")

st.write("Enter the house features below to get an estimated sale price.")

# --- Input Fields ---
overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 5000, 1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Basement Area (Total sq ft)", 0, 3000, 800)
year_built = st.number_input("Year Built", 1870, 2024, 1990)
year_remod = st.number_input("Year Remodeled", 1950, 2024, 2005)
full_bath = st.slider("Full Bathrooms", 0, 3, 2)
bedrooms = st.slider("Bedrooms Above Ground", 1, 5, 3)
fireplaces = st.slider("Fireplaces", 0, 3, 1)
neighborhood = st.selectbox("Neighborhood", [
    'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes',
    'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR',
    'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill',
    'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
])

# --- Model Selection ---
model_choice = st.selectbox("Choose Model", ["XGBoost", "Linear Regression"])
model_key = "xgboost" if model_choice == "XGBoost" else "linear"

# --- Build Input Dict ---
input_data = {
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'GarageCars': garage_cars,
    'TotalBsmtSF': total_bsmt_sf,
    'YearBuilt': year_built,
    'YearRemod/Add': year_remod,
    'FullBath': full_bath,
    'BedroomAbvGr': bedrooms,
    'Fireplaces': fireplaces,
    f'Neighborhood_{neighborhood}': 1  # one-hot encoded field
}

# --- Predict ---
if st.button("Predict Price"):
    try:
        price = predict(input_data, model_type=model_key)
        st.success(f"💰 {model_choice} Predicted Sale Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
