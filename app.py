import streamlit as st
import pickle
import sklearn
import numpy as np
import pandas as pd

background_image = """
<style>
.stApp {
    background-image: url("https://rog.asus.com/media/1610082227143.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    z-index: -1;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
st.title("Laptop Price Prediction")

pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pickle.load(open('models/data.pkl', 'rb'))

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand Name', df['Company'].unique())
    laptop_type = st.selectbox('Laptop Type', df['TypeName'].unique())
    ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input("Laptop Weight(in kgs)")
    touchscreen = st.selectbox('TouchScreen', ['NO', 'Yes'])
    ips = st.selectbox('IPS', ['NO', 'Yes'])

with col2:
    Screen_size = st.number_input('Screen Size(in inches)')
    resolution = st.selectbox('Screen Resolution(in pixels)', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', 
        '3200x1800','2880x1800', '2560x1600', '2560x1440', 
        '2304x1440'
    ])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    if ram == 0 or weight == 0 or ssd == 0 or Screen_size == 0:
        st.error("Please provide valid inputs for all fields")
    else:
        x_res = int(resolution.split('x')[0])
        y_res = int(resolution.split('x')[1])
        ppi = ((x_res**2) + (y_res**2))**0.5 / Screen_size
        
        # Create DataFrame with CORRECT column names from training data
        query_df = pd.DataFrame([[
            company, laptop_type, ram, weight, touchscreen_val, ips_val, ppi, 
            cpu, hdd, ssd, gpu, os
        ]], columns=[
            'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS',
            'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'
        ])
        
        try:
            predicted_log_price = pipe.predict(query_df)
            predicted_price = np.exp(predicted_log_price)
            st.title(f"Predicted Price is ₹ {int(predicted_price[0]):,}")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")