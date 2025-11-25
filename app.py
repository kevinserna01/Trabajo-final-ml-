"""
Sistema de Predicción con Machine Learning
Aplicación web para predicción de churn en telecomunicaciones
y segmentación de clientes de tarjetas de crédito.

Autores: Kevin Serna, Johan Stiven Sinisterra , Juan David Quintero
Proyecto Final - Machine Learning







"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import subprocess



# CONFIGURACIÓN INICIAL DE LA APLICACIÓN
st.set_page_config(
    page_title="ML Models - Churn & Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CARGA DE MODELOS Y RECURSOS
def verificar_y_entrenar_modelos():
    """
    Verifica si los modelos existen. Si no, los entrena automáticamente.
    Útil para el primer despliegue en Streamlit Cloud.
    """
    if not os.path.exists('models/modelo_logistica.pkl'):
        st.warning("Modelos no encontrados. Entrenando modelos por primera vez...")
        st.info("Esto puede tomar 1-2 minutos. Por favor espera...")
        
        # Ejecutar el script de entrenamiento
        try:
            subprocess.run(['python', 'entrenar_modelos.py'], check=True)
            st.success("Modelos entrenados exitosamente!")
            st.rerun()
        except Exception as e:
            st.error(f"Error al entrenar modelos: {str(e)}")
            st.stop()

@st.cache_resource
def cargar_modelos():
    """
    Carga todos los modelos entrenados y recursos necesarios.
    
    Utiliza cache_resource para cargar los modelos una sola vez
    y reutilizarlos en todas las sesiones, mejorando el rendimiento.
    
    Returns:
        tuple: Contiene los modelos, scalers y nombres de columnas
    """
    # Verificar que los modelos existan
    verificar_y_entrenar_modelos()
    
    # Carga de modelos de clasificación
    with open('models/modelo_logistica.pkl', 'rb') as f:
        log_model = pickle.load(f)
    
    with open('models/modelo_knn.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    
    # Carga de modelo de clustering
    with open('models/modelo_kmeans.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    # Carga de scalers (normalizadores de datos)
    with open('scalers/scaler_telco.pkl', 'rb') as f:
        scaler_telco = pickle.load(f)
    
    with open('scalers/scaler_cc.pkl', 'rb') as f:
        scaler_cc = pickle.load(f)
    
    # Carga de nombres de columnas para transformación de datos
    with open('data/columnas_telco.pkl', 'rb') as f:
        columnas_telco = pickle.load(f)
    
    with open('data/columnas_cc.pkl', 'rb') as f:
        columnas_cc = pickle.load(f)
    
    return log_model, knn_model, kmeans_model, scaler_telco, scaler_cc, columnas_telco, columnas_cc


# Inicialización de modelos
log_model, knn_model, kmeans_model, scaler_telco, scaler_cc, columnas_telco, columnas_cc = cargar_modelos()




# FUNCIONES AUXILIARES
def preprocesar_datos_telco(input_data, columnas_esperadas):
    """
    Preprocesa los datos de entrada para modelos de Telco.
    
    Aplica One-Hot Encoding a las variables categóricas y asegura
    que el DataFrame resultante tenga las mismas columnas que los
    datos de entrenamiento.
    
    Args:
        input_data (dict): Diccionario con los datos del formulario
        columnas_esperadas (list): Lista de columnas del modelo entrenado
    
    Returns:
        pd.DataFrame: DataFrame procesado listo para predicción
    """
    # Crear DataFrame desde el diccionario de entrada
    df_input = pd.DataFrame([input_data])
    
    # Identificar columnas categóricas
    categorical_cols = df_input.select_dtypes(include=['object']).columns
    
    # Aplicar One-Hot Encoding (drop_first=True para evitar multicolinealidad)
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
    
    # Agregar columnas faltantes con valor 0
    for col in columnas_esperadas:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reordenar columnas en el mismo orden del entrenamiento
    df_encoded = df_encoded[columnas_esperadas]
    
    return df_encoded


def crear_formulario_telco():
    """
    Crea el formulario de entrada para predicción de Churn.
    
    Returns:
        dict: Diccionario con todos los valores ingresados por el usuario
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Demográficos")
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox(
            "¿Es adulto mayor?", 
            [0, 1], 
            format_func=lambda x: "Sí" if x == 1 else "No"
        )
        partner = st.selectbox("¿Tiene pareja?", ["Yes", "No"])
        dependents = st.selectbox("¿Tiene dependientes?", ["Yes", "No"])
        
        st.subheader("Servicios Básicos")
        phone_service = st.selectbox("Servicio de Teléfono", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Múltiples Líneas", 
            ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox(
            "Servicio de Internet", 
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Seguridad Online", 
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Respaldo Online", 
            ["Yes", "No", "No internet service"]
        )
    
    with col2:
        st.subheader("Servicios Adicionales")
        device_protection = st.selectbox(
            "Protección de Dispositivo", 
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Soporte Técnico", 
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", 
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Películas", 
            ["Yes", "No", "No internet service"]
        )
        
        st.subheader("Información de Contrato")
        contract = st.selectbox(
            "Tipo de Contrato", 
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Facturación sin Papel", ["Yes", "No"])
        payment_method = st.selectbox(
            "Método de Pago",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        
        st.subheader("Información Financiera")
        tenure = st.number_input(
            "Meses de Antigüedad", 
            min_value=0, 
            max_value=72, 
            value=12,
            help="Número de meses que el cliente ha estado con la compañía"
        )
        monthly_charges = st.number_input(
            "Cargo Mensual ($)", 
            min_value=0.0, 
            max_value=150.0, 
            value=50.0,
            step=0.01
        )
        total_charges = st.number_input(
            "Cargo Total ($)", 
            min_value=0.0, 
            max_value=10000.0, 
            value=500.0,
            step=0.01
        )
    
    # Retornar todos los datos como diccionario
    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


# NAVEGACIÓN PRINCIPAl
st.sidebar.title("Navegación")
st.sidebar.markdown("Seleccione el modelo de Machine Learning a utilizar:")

pagina = st.sidebar.radio(
    "Modelos disponibles:", 
    [
        "Inicio",
        "Regresión Logística (Churn)",
        "K-Nearest Neighbors (Churn)",
        "K-Means Clustering"
    ]
)



# PÁGINA DE INICIO
if pagina == "Inicio":
    st.title("Sistema de Predicción con Machine Learning")
    st.markdown("### Proyecto Final - Modelos Supervisados y No Supervisados")
    st.markdown("---")
    
    # Descripción general del proyecto
    st.markdown("""
    Este sistema implementa tres modelos de Machine Learning para resolver
    dos problemas empresariales diferentes:
    
    1. **Predicción de Churn**: Identificar clientes con probabilidad de abandonar el servicio
    2. **Segmentación de Clientes**: Agrupar clientes según su comportamiento financiero
    """)
    
    st.markdown("---")
    
    # Sección de modelos disponibles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Regresión Logística")
        st.write("**Tipo:** Aprendizaje Supervisado")
        st.write("**Objetivo:** Predecir probabilidad de churn")
        st.write("**Salida:** Probabilidad (0-100%) y clasificación binaria")
        st.info("Modelo lineal probabilístico basado en función sigmoide")
    
    with col2:
        st.subheader("K-Nearest Neighbors")
        st.write("**Tipo:** Aprendizaje Supervisado")
        st.write("**Objetivo:** Clasificar clientes por similitud")
        st.write("**Salida:** Clasificación binaria (Churn: Sí/No)")
        st.info("Algoritmo basado en distancia euclidiana (k=9)")
    
    with col3:
        st.subheader("K-Means Clustering")
        st.write("**Tipo:** Aprendizaje No Supervisado")
        st.write("**Objetivo:** Segmentar clientes en grupos")
        st.write("**Salida:** Asignación a cluster (0-3)")
        st.info("Algoritmo de clustering con 4 centroides")
    
    st.markdown("---")
    
    # Métricas de rendimiento
    st.subheader("Métricas de Rendimiento")
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("Regresión Logística AUC", "0.84")
    
    with col_m2:
        st.metric("KNN AUC", "0.82")
    
    with col_m3:
        st.metric("K-Means Silhouette Score", "0.45")
    
    st.markdown("---")
    st.success("Seleccione un modelo en el menú lateral para comenzar el análisis")



# PÁGINA: REGRESIÓN LOGÍSTICA
elif pagina == "Regresión Logística (Churn)":
    st.title("Predicción de Churn - Regresión Logística")
    
    st.markdown("""
    **Descripción del modelo:**
    La regresión logística es un algoritmo de clasificación que utiliza una función
    sigmoide para estimar la probabilidad de que un cliente abandone el servicio.
    
    **Ventajas:**
    - Proporciona probabilidades interpretables
    - Rápido en predicción
    - Buena performance en datos linealmente separables
    """)
    
    st.markdown("---")
    
    # Crear formulario y obtener datos
    input_data = crear_formulario_telco()
    
    # Botón de predicción
    if st.button("Realizar Predicción", type="primary"):
        # Preprocesar datos
        df_encoded = preprocesar_datos_telco(input_data, columnas_telco)
        
        # Realizar predicción
        # predict_proba devuelve [probabilidad_clase_0, probabilidad_clase_1]
        probabilidades = log_model.predict_proba(df_encoded)
        prob_churn = probabilidades[0][1]  # Probabilidad de la clase positiva (Churn=1)
        
        # Clasificación binaria
        prediccion = log_model.predict(df_encoded)[0]
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("Resultados de la Predicción")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.metric(
                "Probabilidad de Churn", 
                f"{prob_churn*100:.2f}%",
                help="Probabilidad de que el cliente abandone el servicio"
            )
        
        with col_r2:
            resultado_texto = "SÍ (Alta probabilidad)" if prediccion == 1 else "NO (Baja probabilidad)"
            st.metric("Predicción Final", resultado_texto)
        
        # Interpretación y recomendaciones
        st.markdown("---")
        st.subheader("Análisis y Recomendaciones")
        
        if prob_churn > 0.7:
            st.error("""
            **RIESGO CRÍTICO**
            - Probabilidad muy alta de abandono
            - Acción inmediata requerida
            - Recomendación: Contactar con oferta personalizada de retención
            """)
        elif prob_churn > 0.5:
            st.warning("""
            **RIESGO MODERADO**
            - Probabilidad significativa de abandono
            - Se recomienda monitoreo cercano
            - Recomendación: Enviar encuesta de satisfacción y evaluar mejoras en el servicio
            """)
        else:
            st.success("""
            **CLIENTE ESTABLE**
            - Baja probabilidad de abandono
            - Cliente satisfecho con el servicio
            - Recomendación: Mantener calidad del servicio y considerar para programas de lealtad
            """)



# PÁGINA: K-NEAREST NEIGHBORS
elif pagina == "K-Nearest Neighbors (Churn)":
    st.title("Predicción de Churn - K-Nearest Neighbors")
    
    st.markdown("""
    **Descripción del modelo:**
    KNN es un algoritmo de clasificación basado en instancias que predice la clase
    de un cliente analizando los k=9 clientes más similares en el conjunto de entrenamiento.
    
    **Ventajas:**
    - No asume distribución de datos
    - Efectivo con patrones no lineales
    - Intuitivo y fácil de interpretar
    """)
    
    st.markdown("---")
    
    # Crear formulario y obtener datos
    input_data = crear_formulario_telco()
    
    # Botón de predicción
    if st.button("Realizar Predicción", type="primary"):
        # Preprocesar datos
        df_encoded = preprocesar_datos_telco(input_data, columnas_telco)
        
        # Realizar predicción
        prediccion = knn_model.predict(df_encoded)[0]
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("Resultados de la Predicción")
        
        resultado_texto = "SÍ - Cliente abandonará el servicio" if prediccion == 1 else "NO - Cliente permanecerá"
        st.metric("Predicción KNN", resultado_texto)
        
        # Interpretación
        st.markdown("---")
        st.subheader("Interpretación del Resultado")
        
        if prediccion == 1:
            st.error("""
            **El modelo KNN predice CHURN**
            
            Basándose en la similitud con otros clientes históricos, este perfil
            muestra características asociadas con alta probabilidad de abandono.
            
            **Acciones recomendadas:**
            - Revisar satisfacción del cliente
            - Ofrecer incentivos de retención
            - Mejorar calidad del servicio
            """)
        else:
            st.success("""
            **El modelo KNN predice NO CHURN**
            
            Este perfil de cliente es similar a aquellos que históricamente
            han permanecido con el servicio.
            
            **Acciones recomendadas:**
            - Mantener la calidad del servicio
            - Considerar para programas de referidos
            - Evaluar oportunidades de upselling
            """)



# PÁGINA: K-MEANS CLUSTERING
elif pagina == "K-Means Clustering":
    st.title("Segmentación de Clientes - K-Means Clustering")
    
    st.markdown("""
    **Descripción del modelo:**
    K-Means es un algoritmo de aprendizaje no supervisado que agrupa clientes
    con comportamiento financiero similar en 4 clusters distintos.
    
    **Aplicación:**
    Permite crear estrategias de marketing personalizadas para cada segmento
    de clientes según sus patrones de uso de tarjeta de crédito.
    """)
    
    st.markdown("---")
    
    # Formulario de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información de Balance y Compras")
        balance = st.number_input(
            "Balance de Tarjeta de Crédito ($)",
            min_value=0.0,
            max_value=20000.0,
            value=1000.0,
            step=10.0,
            help="Balance promedio en la tarjeta de crédito"
        )
        purchases = st.number_input(
            "Compras Totales ($)",
            min_value=0.0,
            max_value=50000.0,
            value=500.0,
            step=10.0,
            help="Total de compras realizadas"
        )
        cash_advance = st.number_input(
            "Avances de Efectivo ($)",
            min_value=0.0,
            max_value=50000.0,
            value=0.0,
            step=10.0,
            help="Total de avances de efectivo solicitados"
        )
        credit_limit = st.number_input(
            "Límite de Crédito ($)",
            min_value=0.0,
            max_value=30000.0,
            value=5000.0,
            step=100.0,
            help="Límite de crédito asignado"
        )
        payments = st.number_input(
            "Pagos Totales ($)",
            min_value=0.0,
            max_value=50000.0,
            value=1000.0,
            step=10.0,
            help="Total de pagos realizados"
        )
    
    with col2:
        st.subheader("Frecuencias de Uso")
        st.markdown("*Valores entre 0 (nunca) y 1 (siempre)*")
        
        balance_freq = st.slider(
            "Frecuencia de Balance Actualizado",
            0.0, 1.0, 0.8,
            help="Frecuencia con la que se actualiza el balance"
        )
        purchases_freq = st.slider(
            "Frecuencia de Compras",
            0.0, 1.0, 0.5,
            help="Frecuencia de realización de compras"
        )
        cash_advance_freq = st.slider(
            "Frecuencia de Avances de Efectivo",
            0.0, 1.0, 0.0,
            help="Frecuencia de solicitud de avances"
        )
        
        st.subheader("Información Adicional")
        minimum_payments = st.number_input(
            "Pagos Mínimos ($)",
            min_value=0.0,
            max_value=20000.0,
            value=200.0,
            step=10.0,
            help="Total de pagos mínimos realizados"
        )
    
    # Botón de predicción
    if st.button("Asignar a Cluster", type="primary"):
        # Crear vector de entrada con todas las características necesarias
        # Nota: Algunas variables se inicializan en 0 por simplicidad
        input_data = {
            'BALANCE': balance,
            'BALANCE_FREQUENCY': balance_freq,
            'PURCHASES': purchases,
            'ONEOFF_PURCHASES': 0,
            'INSTALLMENTS_PURCHASES': 0,
            'CASH_ADVANCE': cash_advance,
            'PURCHASES_FREQUENCY': purchases_freq,
            'ONEOFF_PURCHASES_FREQUENCY': 0,
            'PURCHASES_INSTALLMENTS_FREQUENCY': 0,
            'CASH_ADVANCE_FREQUENCY': cash_advance_freq,
            'CASH_ADVANCE_TRX': 0,
            'PURCHASES_TRX': 0,
            'CREDIT_LIMIT': credit_limit,
            'PAYMENTS': payments,
            'MINIMUM_PAYMENTS': minimum_payments,
            'PRC_FULL_PAYMENT': 0,
            'TENURE': 12
        }
        
        # Crear DataFrame y ordenar columnas
        df_input = pd.DataFrame([input_data])
        df_input = df_input[columnas_cc]
        
        # Aplicar escalado (StandardScaler)
        df_scaled = scaler_cc.transform(df_input)
        
        # Realizar predicción de cluster
        cluster_asignado = kmeans_model.predict(df_scaled)[0]
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("Resultado de Segmentación")
        
        st.metric("Cluster Asignado", f"Grupo {cluster_asignado}")
        
        # Definición de perfiles de clusters
        # Estos perfiles fueron determinados mediante análisis exploratorio
        perfiles = {
            0: {
                "nombre": "Ahorradores / Bajo Uso",
                "descripcion": """
                **Características:**
                - Balance bajo
                - Pocas compras
                - Uso limitado de la tarjeta
                
                **Estrategia de Marketing:**
                - Campañas de activación
                - Incentivos por uso
                - Programas de cashback
                - Educación financiera
                """,
                "tipo": "info"
            },
            1: {
                "nombre": "Gastadores VIP",
                "descripcion": """
                **Características:**
                - Altas compras
                - Buenos pagos
                - Uso frecuente y responsable
                
                **Estrategia de Marketing:**
                - Programas de lealtad premium
                - Ofertas exclusivas
                - Aumento de límite de crédito
                - Servicios adicionales
                """,
                "tipo": "success"
            },
            2: {
                "nombre": "Usuarios de Efectivo",
                "descripcion": """
                **Características:**
                - Altos avances de efectivo
                - Frecuente necesidad de liquidez
                - Uso de tarjeta para cash advance
                
                **Estrategia de Marketing:**
                - Ofrecer préstamos personales
                - Tasas preferenciales
                - Consolidación de deudas
                - Asesoría financiera
                """,
                "tipo": "warning"
            },
            3: {
                "nombre": "Alto Balance / Deudores",
                "descripcion": """
                **Características:**
                - Balance alto mantenido
                - Bajo nivel de gasto
                - Posible acumulación de deuda
                
                **Estrategia de Marketing:**
                - Monitoreo de riesgo crediticio
                - Planes de pago
                - Reestructuración de deuda
                - Prevención de morosidad
                """,
                "tipo": "error"
            }
        }
        
        # Mostrar información del perfil
        perfil = perfiles.get(cluster_asignado, None)
        
        if perfil:
            st.markdown("---")
            st.subheader(f"Perfil: {perfil['nombre']}")
            
            if perfil['tipo'] == "success":
                st.success(perfil['descripcion'])
            elif perfil['tipo'] == "warning":
                st.warning(perfil['descripcion'])
            elif perfil['tipo'] == "error":
                st.error(perfil['descripcion'])
            else:
                st.info(perfil['descripcion'])



# FOOTER DE LA APLICACIÓN
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Información del Proyecto**

Modelos implementados:
- Regresión Logística
- K-Nearest Neighbors
- K-Means Clustering
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 - Proyecto Final Machine Learning")
