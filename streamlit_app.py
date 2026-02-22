import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

st.set_page_config(
    page_title="LankaHome AI | House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model_and_prep():
    model = xgb.XGBRegressor()
    model.load_model('house_price_model.json')
    scaler = joblib.load('scaler.joblib')
    le_dict = joblib.load('encoders.joblib')
    return model, scaler, le_dict

@st.cache_data
def load_source_data():
    return pd.read_csv('house_prices_srilanka.csv')

def main():
    try:
        model, scaler, le_dict = load_model_and_prep()
        df_source = load_source_data()
        
        cols_order = ['district', 'area', 'perch', 'bedrooms', 'bathrooms', 'kitchen_area_sqft', 
                      'parking_spots', 'has_garden', 'has_ac', 'water_supply', 'electricity', 
                      'floors', 'year_built']
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.info("Make sure you have run the training notebook and have the following files: house_price_model.json, scaler.joblib, encoders.joblib, house_prices_srilanka.csv")
        return

    st.title("🏠 LankaHome AI: House Price Predictor")
    st.markdown("Predict house prices across Sri Lanka with high-precision XGBoost machine learning.")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Property Features")
        
        g1, g2 = st.columns(2)
        with g1:
            district = st.selectbox("District", sorted(df_source['district'].unique()))

            filtered_areas = sorted(df_source[df_source['district'] == district]['area'].unique())
            area = st.selectbox("Area", filtered_areas)
            
            perch = st.number_input("Perch (Land Size)", min_value=1.0, value=10.0, step=0.1)
            bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
            floors = st.number_input("Floors", min_value=1, value=2)

        with g2:
            water_supply = st.selectbox("Water Supply", sorted(df_source['water_supply'].unique()))
            electricity = st.selectbox("Electricity", sorted(df_source['electricity'].unique()))
            kitchen_area = st.number_input("Kitchen Area (sqft)", min_value=10, value=150)
            parking = st.number_input("Parking Spots", min_value=0, value=1)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2020)
            
            st.write("Amenities")
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                has_garden = st.checkbox("Has Garden")
            with a_col2:
                has_ac = st.checkbox("Has A/C")

    input_data = {
        'district': district,
        'area': area,
        'perch': perch,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'kitchen_area_sqft': kitchen_area,
        'parking_spots': parking,
        'has_garden': has_garden,
        'has_ac': has_ac,
        'water_supply': water_supply,
        'electricity': electricity,
        'floors': floors,
        'year_built': year_built
    }

    with col2:
        st.header("Outcome")
        predict_btn = st.button("Estimate Market Value", type="primary", use_container_width=True)
        
        if predict_btn:
            with st.spinner("Calculating price..."):
                try:
                    input_df = pd.DataFrame([input_data])
                    
                    for col, le in le_dict.items():
                        if col in input_df.columns:
                            input_df[col] = le.transform(input_df[col].astype(str))

                    input_df = input_df[cols_order]

                    input_scaled = scaler.transform(input_df)
 
                    prediction = model.predict(input_scaled)[0]
                    
                    st.success("Analysis Complete!")
                    st.metric("Estimated Price", f"LKR {prediction:,.2f}")

                    st.info("💡 Tip: Prices can vary based on the specific neighborhood and build quality.")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    st.divider()
    with st.expander("Show Model Insight (Feature Importance)"):
        st.write("This shows which factors are influencing the predictions the most.")
        importance_df = pd.DataFrame({
            'Feature': cols_order,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

if __name__ == "__main__":
    main()
