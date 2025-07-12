import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelo y preprocesador
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Columnas esperadas
expected_columns = {
    'totrms_abvgrd', 'lot_config', 'bsmtfin_sf_2', 'quality_sf', 'open_porch_sf',
    'lot_shape', 'overall_qual', 'total_bsmt_sf', 'full_bath', 'exter_cond',
    'garage_cars', 'fireplaces', 'paved_drive', 'low_qual_fin_sf', 'bsmt_unf_sf',
    'house_style', '2nd_flr_sf', 'unnamed_0', '1st_flr_sf', 'bsmtfin_sf_1',
    'ms_subclass', 'bsmtfin_type_2', 'central_air', 'bsmt_cond', 'year_built',
    'garage_type', 'heating_qc', 'electrical', 'roof_style', 'pool_area',
    'screen_porch', 'garage_cond', 'condition_2', 'neighborhood', '3ssn_porch',
    'garage_qual', 'garage_finish', 'functional', 'garage_area', 'fireplace_qu',
    'enclosed_porch', 'mo_sold', 'total_bedrooms', 'land_contour', 'bsmt_full_bath',
    'lot_frontage', 'land_slope', 'bsmt_exposure', 'bsmt_half_bath', 'finished_sf',
    'utilities', 'sale_condition', 'condition_1', 'bedroom_abvgr', 'total_bath',
    'garage_yr_blt', 'bsmt_qual', 'misc_val', 'bldg_type', 'roof_matl', 'foundation',
    'year_remodadd', 'exterior_2nd', 'kitchen_abvgr', 'overall_cond', 'total_sf',
    'exter_qual', 'exterior_1st', 'gr_liv_area', 'kitchen_qual', 'heating',
    'bsmtfin_type_1', 'mas_vnr_area', 'wood_deck_sf', 'pid', 'sale_type', 'half_bath',
    'lot_area', 'yr_sold', 'street', 'ms_zoning'
}

# Campos categóricos (esto puede variar si cambias el dataset)
categorical_columns = {
    'lot_config', 'lot_shape', 'exter_cond', 'house_style', 'bsmtfin_type_2', 'central_air',
    'bsmt_cond', 'garage_type', 'heating_qc', 'electrical', 'roof_style', 'paved_drive',
    'condition_2', 'neighborhood', 'garage_qual', 'garage_finish', 'functional',
    'fireplace_qu', 'land_contour', 'bsmt_exposure', 'utilities', 'sale_condition',
    'condition_1', 'bldg_type', 'roof_matl', 'foundation', 'exterior_2nd', 'exterior_1st',
    'exter_qual', 'kitchen_qual', 'heating', 'bsmtfin_type_1', 'sale_type',
    'ms_zoning', 'street'
}

# Interfaz de entrada
st.title("🏡 Predicción del Precio de Viviendas")
st.write("Ingrese los datos de una vivienda para predecir su precio")

# Inputs reales
lot_area = st.number_input("Lot Area", value=8000)
year_sold = st.selectbox("Año de venta", [2006, 2007, 2008, 2009, 2010])
street = st.selectbox("Tipo de calle", ['Pave', 'Grvl'])
ms_zoning = st.selectbox("MS Zoning", ['RL', 'RM', 'FV', 'RH', 'C (all)'])

# Crear DataFrame base
input_df = pd.DataFrame({
    'lot_area': [lot_area],
    'yr_sold': [year_sold],
    'street': [street],
    'ms_zoning': [ms_zoning]
})

# Completar columnas faltantes
for col in expected_columns:
    if col not in input_df.columns:
        if col in categorical_columns:
            input_df[col] = 'Desconocido'
        else:
            input_df[col] = np.nan  # Mejor que '0' para columnas que usan imputación

# Reordenar
input_df = input_df[list(expected_columns)]

# Transformar y predecir
input_processed = preprocessor.transform(input_df)
prediction = model.predict(input_processed)[0]

st.success(f"💰 Precio estimado de la vivienda: ${prediction:,.2f}")
