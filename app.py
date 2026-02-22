from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'house_price_model.json'
SCALER_PATH = 'scaler.joblib'
ENCODERS_PATH = 'encoders.joblib'

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le_dict = joblib.load(ENCODERS_PATH)

df_source = pd.read_csv('house_prices_srilanka.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/options', methods=['GET'])
def get_options():
    districts = sorted(df_source['district'].unique().tolist())

    areas_by_district = {d: sorted(df_source[df_source['district'] == d]['area'].unique().tolist()) for d in districts}
    
    return jsonify({
        'districts': districts,
        'areas_by_district': areas_by_district,
        'water_supply': sorted(df_source['water_supply'].unique().tolist()),
        'electricity': sorted(df_source['electricity'].unique().tolist()),
        'has_garden': [False, True],
        'has_ac': [False, True]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
        for col, le in le_dict.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))
        
        cols_order = ['district', 'area', 'perch', 'bedrooms', 'bathrooms', 'kitchen_area_sqft', 
                      'parking_spots', 'has_garden', 'has_ac', 'water_supply', 'electricity', 
                      'floors', 'year_built']
        input_df = input_df[cols_order]
        
        input_scaled = scaler.transform(input_df)
        
        prediction = model.predict(input_scaled)
        
        return jsonify({
            'status': 'success',
            'prediction': float(prediction[0]),
            'formatted_prediction': f"LKR {prediction[0]:,.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
