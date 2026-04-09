import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la interfaz
st.set_page_config(page_title="Predicción de Viabilidad", layout="centered")

@st.cache_resource
def load_assets():
    # Cargamos tu archivo .pkl tal cual lo tienes
    with open('modelo-class.pkl', 'rb') as f:
        # El orden según tu estructura es: [modelo, encoder, variables, scaler]
        data = joblib.load(f)
    return {
        "model": data[0],
        "variables": data[2],
        "scaler": data[3]
    }

assets = load_assets()
model = assets["model"]
variables_modelo = assets["variables"]
scaler = assets["scaler"]

st.title("🏗️ Evaluación de Viabilidad")
st.markdown("Ingrese los datos para predecir la viabilidad del proyecto.")

with st.form("form_pred"):
    col1, col2 = st.columns(2)
    with col1:
        precio = st.number_input("Precio por m² (PRECIOVTAX)", value=4200.0)
        estrato = st.selectbox("Estrato (ESTRATO)", [1, 2, 3, 4, 5, 6], index=3)
        area = st.number_input("Área Total (AREATOTZC)", value=200.0)
    with col2:
        avance = st.slider("Avance de Obra (GRADOAVANC)", 0, 100, 10)
        pisos = st.number_input("Número de Pisos (NRO_PISOS)", value=5)
        tipo_val = st.selectbox("Tipo de Valor (TIPOVRDEST)", [1, 2], format_func=lambda x: "Real" if x==1 else "Estimado")
    
    submit = st.form_submit_button("ANALIZAR")

if submit:
    try:
        # 1. Crear un DataFrame vacío con las columnas EXACTAS que el scaler espera
        # Esto soluciona el error de USO_DOS_3, ya que ahora la columna existirá
        input_df = pd.DataFrame(np.zeros((1, len(variables_modelo))), columns=variables_modelo)
        
        # 2. Asignar los valores de la interfaz a las columnas correspondientes
        # Usamos los nombres técnicos que el modelo tiene en su memoria
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
        
        # 3. Reordenar las columnas para que coincidan 100% con el fit original
        input_df = input_df[variables_modelo]

        # 4. Escalar y Predecir
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # 5. Mostrar resultados
        st.divider()
        if pred == 1:
            st.success(f"✅ PROYECTO VIABLE (Confianza: {prob[1]*100:.1f}%)")
        else:
            st.error(f"❌ PROYECTO NO VIABLE (Confianza: {prob[0]*100:.1f}%)")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        with st.expander("Ver diagnóstico"):
            st.write("Columnas esperadas por el modelo:", variables_modelo)
