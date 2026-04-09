import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Viabilidad de Tokenización Inmobiliaria",
    page_icon="🏗️",
    layout="centered"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main { background-color: #f5f4f0; }
    .stButton>button { background-color: #1a3a2a; color: white; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE ACTIVOS ---
@st.cache_resource
def load_model_assets():
    # Cargamos el archivo que contiene el modelo, scaler y variables
    # Asegúrate de que el nombre sea exacto al de tu archivo
    with open('modelo-class.pkl', 'rb') as f:
        data = joblib.load(f)
    
    # Dependiendo de cómo guardaste el pkl, extraemos los componentes
    # Basado en tu PDF, el orden suele ser: modelo, encoder, variables, scaler
    model = data[0]
    variables = data[2]
    scaler = data[3]
    return model, variables, scaler

try:
    modelNN, variables, scaler = load_model_assets()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Verifica que 'modelo-class.pkl' esté en la raíz del repositorio.")
    st.stop()

# --- INTERFAZ ---
st.title("🏗️ Viabilidad de Tokenización Inmobiliaria")
st.markdown("**Antioquia, Colombia** | Modelo de clasificación basado en datos CEED-DANE")
st.divider()

st.markdown("### Ingrese los datos del proyecto")

with st.form("form_viabilidad"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Ubicación y Perfil")
        estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6], index=3)
        area_total = st.number_input("Área Total (m²)", value=500.0)
        nro_pisos = st.number_input("Número de Pisos", value=5, min_value=1)
        
    with col2:
        st.subheader("💰 Financiero")
        precio_vtax = st.number_input("Precio por m² (mil COP)", value=4500.0)
        tipo_valor = st.selectbox("Tipo de Valor", [1, 2], format_func=lambda x: "Real" if x==1 else "Estimado")
        grado_avance = st.slider("Grado de Avance (%)", 0, 100, 10)

    predecir = st.form_submit_button("EVALUAR VIABILIDAD")

if predecir:
    # 1. Crear el vector de entrada con todas las columnas que el modelo espera
    # Creamos un DataFrame con ceros para las columnas que no pedimos manualmente
    input_df = pd.DataFrame(np.zeros((1, len(variables))), columns=variables)
    
    # 2. Mapear los inputs del formulario a las columnas del DataFrame
    # Usamos los nombres exactos que extrajiste en tu proceso de PCA/Selección
    mapping = {
        'ESTRATO': estrato,
        'AREATOTZC': area_total,
        'PRECIOVTAX': precio_vtax,
        'GRADOAVANC': grado_avance,
        'NRO_PISOS': nro_pisos,
        'TIPOVRDEST': tipo_valor
    }
    
    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    # 3. Escalar los datos
    input_scaled = scaler.transform(input_df)
    
    # 4. Predicción
    prediccion = modelNN.predict(input_scaled)[0]
    probabilidades = modelNN.predict_proba(input_scaled)[0]

    # --- MOSTRAR RESULTADOS ---
    st.divider()
    if prediccion == 1:
        st.success("✅ **PROYECTO VIABLE** para Tokenización")
        st.metric("Confianza del Modelo", f"{probabilidades[1]*100:.1f}%")
        st.balloons()
    else:
        st.error("❌ **PROYECTO NO VIABLE** según parámetros actuales")
        st.metric("Confianza del Modelo", f"{probabilidades[0]*100:.1f}%")

    st.markdown("### Análisis de Riesgo")
    if estrato <= 2:
        st.warning("⚠️ **Alerta:** El estrato socioeconómico detectado presenta una alta correlación con proyectos de baja rotación financiera.")
    if grado_avance < 15:
        st.info("ℹ️ **Nota:** La etapa temprana del proyecto incrementa la incertidumbre operativa.")

st.divider()
st.caption("Investigación Seminario: El Nuevo Paradigma Inmobiliario - Antioquia 2026.")