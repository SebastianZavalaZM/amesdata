import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y preprocesador
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

st.title("🏡 Predicción del Precio de Viviendas")
st.write("Ingrese los datos de una vivienda para predecir su precio")

# Campos de entrada personalizados
lot_area = st.number_input("Lot Area", value=8000)
year_sold = st.selectbox("Año de venta", [2006, 2007, 2008, 2009, 2010])
street = st.selectbox("Tipo de calle", ['Pave', 'Grvl'])
ms_zoning = st.selectbox("MS Zoning", ['RL', 'RM', 'FV', 'RH', 'C (all)'])

# DataFrame con los datos ingresados
input_df = pd.DataFrame({
    'lot_area': [lot_area],
    'yr_sold': [year_sold],
    'street': [street],
    'ms_zoning': [ms_zoning]
    # Agrega más columnas si tu preprocesador las necesita
})

# Preprocesar e inferir
input_processed = preprocessor.transform(input_df)
prediction = model.predict(input_processed)[0]
st.success(f"💰 Precio estimado de la vivienda: ${prediction:,.2f}")
