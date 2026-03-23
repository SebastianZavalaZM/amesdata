# 🏡 Predicción del Precio de Viviendas

Una aplicación web desarrollada con Streamlit para predecir el precio de viviendas utilizando machine learning.

## Características

- Interfaz intuitiva y fácil de usar
- Predicción en tiempo real
- Modelo de machine learning pre-entrenado
- Variables de entrada personalizables

## Requisitos

- Python 3.7+
- Streamlit
- Pandas
- Joblib

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/SebastianZavalaZM/amesdata.git
cd amesdata
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecuta la aplicación:
```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## Variables de entrada

- **Lot Area**: Área del lote
- **Año de venta**: Año en que se vendió la propiedad
- **Tipo de calle**: Tipo de acceso vial (Pavimentado/Grava)
- **MS Zoning**: Zonificación municipal
