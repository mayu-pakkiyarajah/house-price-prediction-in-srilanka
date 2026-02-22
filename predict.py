import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import sys

def predict_house_price():
    MODEL_PATH = 'house_price_model.json'
    SCALER_PATH = 'scaler.joblib'
    ENCODERS_PATH = 'encoders.joblib'

    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH]):
        print("Error: Model or preprocessing files not found. Run the notebook first.")
        return

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_dict = joblib.load(ENCODERS_PATH)

    print("\n--- Sri Lanka House Price Predictor (Terminal) ---")
    print("Please enter the following details:")

    try:
        data = {
            'district': input("District (e.g. Colombo): "),
            'area': input("Area (e.g. Borella): "),
            'perch': float(input("Perch: ")),
            'bedrooms': int(input("Bedrooms: ")),
            'bathrooms': int(input("Bathrooms: ")),
            'kitchen_area_sqft': int(input("Kitchen Area (sqft): ")),
            'parking_spots': int(input("Parking Spots: ")),
            'has_garden': input("Has Garden? (y/n): ").lower() == 'y',
            'has_ac': input("Has A/C? (y/n): ").lower() == 'y',
            'water_supply': input("Water Supply (e.g. Pipe-borne): "),
            'electricity': input("Electricity (e.g. Single phase): "),
            'floors': int(input("Floors: ")),
            'year_built': int(input("Year Built: "))
        }

        input_df = pd.DataFrame([data])
        
        for col, le in le_dict.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        cols_order = ['district', 'area', 'perch', 'bedrooms', 'bathrooms', 'kitchen_area_sqft', 
                      'parking_spots', 'has_garden', 'has_ac', 'water_supply', 'electricity', 
                      'floors', 'year_built']
        input_df = input_df[cols_order]
        
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        
        print(f"\n======================================")
        print(f" ESTIMATED PRICE: LKR {prediction:,.2f}")
        print(f"======================================\n")

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import os
    predict_house_price()
