import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(page_title="Predicción de Viabilidad", layout="centered")

@st.cache_resource
def load_assets():
    # Cargamos el archivo tal como lo tienes
    with open('modelo-class.pkl', 'rb') as f:
        # Según tu error, el pkl es una lista: [modelo, encoder, variables, scaler]
        data = joblib.load(f)
    
    # Extraemos los componentes
    model = data[0]
    variables_entrenamiento = data[2] # Esta es la lista que causó el error
    scaler = data[3]
    return model, variables_entrenamiento, scaler

model, variables, scaler = load_assets()

st.title("🏗️ Predicción de Viabilidad")

# --- FORMULARIO ---
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        # Asegúrate de usar los nombres exactos que espera tu modelo
        precio = st.number_input("Precio por m² (PRECIOVTAX)", value=2500.0)
        estrato = st.selectbox("Estrato (ESTRATO)", [1,2,3,4,5,6], index=3)
        area = st.number_input("Área Total (AREATOTZC)", value=150.0)
    with col2:
        avance = st.slider("Grado de Avance (GRADOAVANC)", 0, 100, 10)
        pisos = st.number_input("Pisos (NRO_PISOS)", value=1)
        tipo_val = st.selectbox("Tipo de Valor (TIPOVRDEST)", [1, 2], format_func=lambda x: "Real" if x==1 else "Estimado")
    
    predict_btn = st.form_submit_button("Evaluar Viabilidad")

if predict_btn:
    try:
        # PASO CLAVE: Crear un DataFrame con TODAS las columnas que el modelo conoce
        # Llenamos todo con ceros inicialmente
        input_df = pd.DataFrame(np.zeros((1, len(variables))), columns=variables)
        
        # PASO CLAVE: Asignar los valores de la interfaz a las columnas correctas
        # Aquí debes usar los nombres EXACTOS que aparecen en tu base de datos
        input_df['PRECIOVTAX'] = precio
        input_df['ESTRATO'] = estrato
        input_df['AREATOTZC'] = area
        input_df['GRADOAVANC'] = avance
        input_df['NRO_PISOS'] = pisos
        input_df['TIPOVRDEST'] = tipo_val
        
        # PASO CLAVE: Forzar el orden de las columnas
        # Esto reordena el DataFrame para que coincida con el scaler exactamente
        input_df = input_df[variables]

        # Escalar y Predecir
        X_scaled = scaler.transform(input_df)
        prediction = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        if prediction == 1:
            st.success(f"✅ PROYECTO VIABLE (Probabilidad: {prob[1]*100:.1f}%)")
        else:
            st.error(f"❌ PROYECTO NO VIABLE (Probabilidad: {prob[0]*100:.1f}%)")

    except Exception as e:
        st.error(f"Error en el proceso: {e}")
        # Si falla, mostramos qué columnas espera el modelo para que las verifiques
        with st.expander("Ver variables esperadas por el modelo"):
            st.write(variables)
