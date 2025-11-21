"""
Aplicación Web para Predicción de Churn y Segmentación de Clientes
Desarrollado con Streamlit
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="ML Models - Churn & Clustering",
    page_icon="🤖",
    layout="wide"
)

# Cargar modelos y scalers
@st.cache_resource
def cargar_modelos():
    with open('modelo_logistica.pkl', 'rb') as f:
        log_model = pickle.load(f)
    with open('modelo_knn.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('modelo_kmeans.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('scaler_telco.pkl', 'rb') as f:
        scaler_telco = pickle.load(f)
    with open('scaler_cc.pkl', 'rb') as f:
        scaler_cc = pickle.load(f)
    with open('columnas_telco.pkl', 'rb') as f:
        columnas_telco = pickle.load(f)
    with open('columnas_cc.pkl', 'rb') as f:
        columnas_cc = pickle.load(f)
    
    return log_model, knn_model, kmeans_model, scaler_telco, scaler_cc, columnas_telco, columnas_cc

log_model, knn_model, kmeans_model, scaler_telco, scaler_cc, columnas_telco, columnas_cc = cargar_modelos()

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")
pagina = st.sidebar.radio("Selecciona un modelo:", 
                          ["🏠 Inicio", 
                           "📊 Regresión Logística (Churn)", 
                           "🔍 KNN (Churn)", 
                           "💳 K-Means (Clustering)"])

# ============================================
# PÁGINA DE INICIO
# ============================================
if pagina == "🏠 Inicio":
    st.title("🤖 Sistema de Predicción con Machine Learning")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Regresión Logística")
        st.write("Predice si un cliente abandonará el servicio de telecomunicaciones.")
        st.info("**Target:** Churn (Yes/No)")
    
    with col2:
        st.subheader("🔍 K-Nearest Neighbors")
        st.write("Clasifica clientes según sus vecinos más cercanos.")
        st.info("**Algoritmo:** Basado en distancia")
    
    with col3:
        st.subheader("💳 K-Means Clustering")
        st.write("Agrupa clientes de tarjetas de crédito en perfiles.")
        st.info("**Grupos:** 4 clusters")
    
    st.markdown("---")
    st.success("👈 Selecciona un modelo en el menú lateral para comenzar.")

# ============================================
# PÁGINA REGRESIÓN LOGÍSTICA
# ============================================
elif pagina == "📊 Regresión Logística (Churn)":
    st.title("📊 Predicción de Churn - Regresión Logística")
    st.write("Ingresa los datos del cliente para predecir si abandonará el servicio.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Demográficos")
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox("¿Es adulto mayor?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        
        st.subheader("Servicios")
        phone_service = st.selectbox("Servicio de Teléfono", ["Yes", "No"])
        multiple_lines = st.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Respaldo Online", ["Yes", "No", "No internet service"])
    
    with col2:
        st.subheader("Servicios Adicionales")
        device_protection = st.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Películas", ["Yes", "No", "No internet service"])
        
        st.subheader("Contrato y Pagos")
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Facturación sin Papel", ["Yes", "No"])
        payment_method = st.selectbox("Método de Pago", 
                                      ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.subheader("Datos Financieros")
        tenure = st.number_input("Meses de Antigüedad", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Cargo Mensual ($)", min_value=0.0, max_value=150.0, value=50.0)
        total_charges = st.number_input("Cargo Total ($)", min_value=0.0, max_value=10000.0, value=500.0)
    
    if st.button("🔮 Predecir con Regresión Logística", type="primary"):
        # Crear dataframe con los datos del formulario
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        df_input = pd.DataFrame([input_data])
        
        # One-Hot Encoding (igual que en el entrenamiento)
        categorical_cols = df_input.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
        
        # Asegurarse de que tenga las mismas columnas que en entrenamiento
        for col in columnas_telco:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[columnas_telco]
        
        # Predicción
        prob = log_model.predict_proba(df_encoded)[0][1]
        pred = log_model.predict(df_encoded)[0]
        
        st.markdown("---")
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.metric("Probabilidad de Churn", f"{prob*100:.2f}%")
        
        with col_r2:
            resultado = "🔴 SÍ" if pred == 1 else "🟢 NO"
            st.metric("Predicción", resultado)
        
        if prob > 0.5:
            st.error("⚠️ Cliente en riesgo alto de abandono. Acción recomendada: Contactar con oferta de retención.")
        else:
            st.success("✅ Cliente estable. Probabilidad baja de abandono.")

# ============================================
# PÁGINA KNN
# ============================================
elif pagina == "🔍 KNN (Churn)":
    st.title("🔍 Predicción de Churn - K-Nearest Neighbors")
    st.write("Usa exactamente el mismo formulario que Regresión Logística.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Demográficos")
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox("¿Es adulto mayor?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        
        st.subheader("Servicios")
        phone_service = st.selectbox("Servicio de Teléfono", ["Yes", "No"])
        multiple_lines = st.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Respaldo Online", ["Yes", "No", "No internet service"])
    
    with col2:
        st.subheader("Servicios Adicionales")
        device_protection = st.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Películas", ["Yes", "No", "No internet service"])
        
        st.subheader("Contrato y Pagos")
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Facturación sin Papel", ["Yes", "No"])
        payment_method = st.selectbox("Método de Pago", 
                                      ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        st.subheader("Datos Financieros")
        tenure = st.number_input("Meses de Antigüedad", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Cargo Mensual ($)", min_value=0.0, max_value=150.0, value=50.0)
        total_charges = st.number_input("Cargo Total ($)", min_value=0.0, max_value=10000.0, value=500.0)
    
    if st.button("🔮 Predecir con KNN", type="primary"):
        input_data = {
            'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
            'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        
        df_input = pd.DataFrame([input_data])
        categorical_cols = df_input.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
        
        for col in columnas_telco:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[columnas_telco]
        
        pred = knn_model.predict(df_encoded)[0]
        
        st.markdown("---")
        resultado = "🔴 SÍ" if pred == 1 else "🟢 NO"
        st.metric("Predicción KNN", resultado)
        
        if pred == 1:
            st.error("⚠️ El modelo KNN predice que el cliente abandonará el servicio.")
        else:
            st.success("✅ El modelo KNN predice que el cliente permanecerá.")

# ============================================
# PÁGINA K-MEANS
# ============================================
elif pagina == "💳 K-Means (Clustering)":
    st.title("💳 Segmentación de Clientes - K-Means")
    st.write("Ingresa los datos financieros del cliente de tarjeta de crédito.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Balance y Compras")
        balance = st.number_input("Balance", min_value=0.0, max_value=20000.0, value=1000.0)
        purchases = st.number_input("Compras Totales", min_value=0.0, max_value=50000.0, value=500.0)
        cash_advance = st.number_input("Avances de Efectivo", min_value=0.0, max_value=50000.0, value=0.0)
        credit_limit = st.number_input("Límite de Crédito", min_value=0.0, max_value=30000.0, value=5000.0)
        payments = st.number_input("Pagos Totales", min_value=0.0, max_value=50000.0, value=1000.0)
    
    with col2:
        st.subheader("Frecuencias (0 a 1)")
        balance_freq = st.slider("Frecuencia de Balance", 0.0, 1.0, 0.8)
        purchases_freq = st.slider("Frecuencia de Compras", 0.0, 1.0, 0.5)
        cash_advance_freq = st.slider("Frecuencia de Avances", 0.0, 1.0, 0.0)
        
        st.subheader("Otros")
        minimum_payments = st.number_input("Pagos Mínimos", min_value=0.0, max_value=20000.0, value=200.0)
    
    if st.button("🔮 Asignar Cluster", type="primary"):
        # Crear vector de entrada (debe coincidir con las columnas del entrenamiento)
        input_data = {
            'BALANCE': balance, 'BALANCE_FREQUENCY': balance_freq, 'PURCHASES': purchases,
            'ONEOFF_PURCHASES': 0, 'INSTALLMENTS_PURCHASES': 0, 'CASH_ADVANCE': cash_advance,
            'PURCHASES_FREQUENCY': purchases_freq, 'ONEOFF_PURCHASES_FREQUENCY': 0,
            'PURCHASES_INSTALLMENTS_FREQUENCY': 0, 'CASH_ADVANCE_FREQUENCY': cash_advance_freq,
            'CASH_ADVANCE_TRX': 0, 'PURCHASES_TRX': 0, 'CREDIT_LIMIT': credit_limit,
            'PAYMENTS': payments, 'MINIMUM_PAYMENTS': minimum_payments, 'PRC_FULL_PAYMENT': 0, 'TENURE': 12
        }
        
        df_input = pd.DataFrame([input_data])
        df_input = df_input[columnas_cc]
        
        df_scaled = scaler_cc.transform(df_input)
        cluster = kmeans_model.predict(df_scaled)[0]
        
        st.markdown("---")
        st.metric("Cluster Asignado", f"Grupo {cluster}")
        
        # Descripciones de clusters
        perfiles = {
            0: "🟢 **Ahorradores / Bajo Uso**: Clientes con bajo balance y pocas compras. Estrategia: Activación.",
            1: "🔵 **Gastadores VIP**: Altas compras y buenos pagos. Estrategia: Programas de lealtad.",
            2: "🟡 **Usuarios de Efectivo**: Altos avances de efectivo. Estrategia: Ofrecer préstamos.",
            3: "🔴 **Alto Balance / Deudores**: Balance alto, bajo gasto. Estrategia: Monitoreo de riesgo."
        }
        
        st.info(perfiles.get(cluster, "Perfil no definido"))

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado con Streamlit 🚀")

