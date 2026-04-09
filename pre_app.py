import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Viabilidad de Tokenización", layout="centered")

@st.cache_resource
def load_assets():
    with open('modelo-class.pkl', 'rb') as f:
        # El orden según tu archivo es: [modelo, encoder, variables, scaler]
        data = joblib.load(f)
    return {
        "model": data[0],
        "variables": data[2],
        "scaler": data[3]
    }

try:
    assets = load_assets()
    model = assets["model"]
    variables = assets["variables"] # Aquí está la lista que contiene 'USO_DOS_3'
    scaler = assets["scaler"]
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

st.title("🏗️ Evaluación de Viabilidad")
st.markdown("Ingrese los datos base para determinar la viabilidad del proyecto.")

with st.form("form_pred"):
    col1, col2 = st.columns(2)
    with col1:
        precio = st.number_input("Precio por m² (mil COP)", value=4200.0)
        estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6], index=3)
        area = st.number_input("Área Total (m²)", value=200.0)
    with col2:
        avance = st.slider("Avance de Obra (%)", 0, 100, 10)
        pisos = st.number_input("Número de Pisos", value=5)
        tipo_val = st.selectbox("Tipo de Valor", [1, 2], format_func=lambda x: "Real" if x==1 else "Estimado")
    
    submit = st.form_submit_button("ANALIZAR VIABILIDAD")

if submit:
    try:
        # 1. Crear DataFrame con Ceros usando la estructura EXACTA del entrenamiento
        # Esto garantiza que 'USO_DOS_3' y otras variables existan con valor 0
        input_df = pd.DataFrame(np.zeros((1, len(variables))), columns=variables)
        
        # 2. Mapear solo los campos que tenemos en la interfaz
        # Usamos nombres estándar del CEED-DANE que suelen estar en tu modelo
        mapping = {
            'PRECIOVTAX': precio,
            'ESTRATO': estrato,
            'AREATOTZC': area,
            'GRADOAVANC': avance,
            'NRO_PISOS': pisos,
            'TIPOVRDEST': tipo_val
        }
        
        for col, val in mapping.items():
            if col in input_df.columns:
                input_df[col] = val
        
        # 3. Asegurar que el orden sea IDÉNTICO al que espera el scaler
        input_df = input_df[variables]

        # 4. Procesamiento
        input_scaled = scaler.transform(input_df)
        prediccion = model.predict(input_scaled)[0]
        probabilidades = model.predict_proba(input_scaled)[0]

        # 5. Visualización de resultados
        st.divider()
        if prediccion == 1:
            st.success(f"✅ PROYECTO VIABLE (Confianza: {probabilidades[1]*100:.1f}%)")
        else:
            st.error(f"❌ PROYECTO NO VIABLE (Confianza: {probabilidades[0]*100:.1f}%)")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        # En caso de error, mostramos las variables para diagnóstico
        with st.expander("Ver diagnóstico de variables"):
            st.write("Variables esperadas por el modelo:", variables)
