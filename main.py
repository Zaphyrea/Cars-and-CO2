from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

def main():
    data_model = pd.read_csv("CO2_cars/autres/co2-voiture/data_model.csv")
    
    # Get min and max values for the mass sliders
    minmin, minmax, maxmin, maxmax = data_model['masse_ordma_min'].min(), data_model['masse_ordma_min'].max(), data_model['masse_ordma_max'].min(), data_model['masse_ordma_max'].max()
    
    # Load pre-trained machine learning models and encoders using joblib    
    model = load('CO2_cars/autres/co2-voiture/Model.pkl')
    encoder = load('CO2_cars/autres/co2-voiture/Encoder.pkl')
    scaler = load('CO2_cars/autres/co2-voiture/Scaler.pkl')
    
    # Set up the title of the Streamlit app
    st.title('Pick a car and get the CO2 emission')

    # Get unique carrosserie values from the loaded data  (to use in the dropdown)  
    carrosserie_data = data_model['Carrosserie'].unique()
    
    with st.form("my_form"):
        # Create a dropdown to choose carrosserie
        carrosserie = st.selectbox('Choose a car\'s body', carrosserie_data)
       
        # Create sliders to select mass values        
        masse_min = st.slider("Minimum Admissible Mass", min_value=minmin, max_value=minmax, value=minmin)
        masse_max = st.slider("Maximum Authorized Mass", min_value=maxmin, max_value=maxmax, value=maxmin)
        # masse_min = st.selectbox("Masse Minimale Admissible", [minmin, minmax], index=0)
        # masse_max = st.selectbox("Masse Maximale Autorisée", [maxmin, maxmax], index=0)
                
        # Create a button to submit the form
        submitted = st.form_submit_button("Predict CO2")
        
        if submitted:
            carrosserie = encoder.transform([carrosserie])
            # Create an input array for the machine learning model            
            X = np.array([carrosserie[0], masse_min, masse_max])
            X = X.reshape(1, -1)
            # Scale the input data using the loaded scaler
            X = scaler.transform(X)
            # Make a CO2 prediction using the loaded model
            y_predict = model.predict(X).round(2)           
            # Display the CO2 prediction and the predicted probability
            st.title("CO2: " + str(y_predict[0]))
            
# To launch the app
if __name__ == "__main__":
    main()
