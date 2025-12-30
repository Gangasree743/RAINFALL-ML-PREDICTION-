import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="🌧️ Rainfall Prediction", layout="wide")

st.title("🌧️ Rainfall Prediction App")
st.markdown("Predict rainfall using Machine Learning")

# Load models and scaler
try:
    rf_model = joblib.load('RF_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("❌ Models not found!")
except Exception as e:
    model_loaded = False
    st.error(f"❌ Error loading models: {str(e)}")
    st.markdown("""
    ### 🔴 Action Required:
    
    The trained models are missing. Please follow these steps:
    
    1. **Open Terminal/Command Prompt** in this folder
    2. **Run the training script:**
       ```
       python train_model.py
       ```
    3. **Wait for completion** - You should see ✅ "Training Complete!"
    4. **Then run Streamlit:**
       ```
       streamlit run app.py
       ```
    
    This will train and save the models, then launch the app!
    """)
    st.stop()

if model_loaded:
    # Sidebar for input
    st.sidebar.header("📊 Input Features")
    st.sidebar.markdown("---")
    
    # Get input from user - these are the top 5 features selected by SelectKBest
    input_data = {}
    
    input_data['MinTemp'] = st.sidebar.slider("Min Temperature (°C)", -10.0, 30.0, 10.0)
    input_data['MaxTemp'] = st.sidebar.slider("Max Temperature (°C)", 0.0, 50.0, 25.0)
    input_data['Rainfall'] = st.sidebar.slider("Rainfall (mm)", 0.0, 100.0, 10.0)
    input_data['Evaporation'] = st.sidebar.slider("Evaporation (mm)", 0.0, 20.0, 5.0)
    input_data['Sunshine'] = st.sidebar.slider("Sunshine (hours)", 0.0, 14.0, 8.0)
    
    # Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Scale the input
    df_scaled = scaler.transform(df_input)
    
    # Make prediction
    if st.sidebar.button("🔮 Predict Rainfall", key="predict_btn"):
        prediction = rf_model.predict(df_scaled)[0]
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Prediction Result")
            st.metric(
                label="Predicted Rainfall",
                value=f"{prediction:.2f} mm",
                delta=None
            )
        
        with col2:
            st.subheader("📥 Input Summary")
            for key, value in input_data.items():
                st.write(f"**{key}:** {value}")
        
        # Additional insights
        st.markdown("---")
        st.subheader("💡 Rainfall Interpretation")
        
        if prediction < 1:
            st.success("🟢 **No Rain Expected** - Rainfall will be very minimal")
        elif prediction < 10:
            st.info("🟡 **Light Rain** - Expect light rainfall")
        elif prediction < 25:
            st.warning("🟠 **Moderate Rain** - Expect moderate rainfall")
        else:
            st.error("🔴 **Heavy Rain** - Expect heavy rainfall, be cautious!")
    
    # Model Information
    st.markdown("---")
    with st.expander("ℹ️ Model Information"):
        st.write("""
        - **Model Type:** Random Forest Regression
        - **Number of Trees:** 5
        - **Input Features:** 5 selected features (MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine)
        - **Target Variable:** Rainfall (mm)
        - **Data Source:** weatherAUS.csv
        """)
