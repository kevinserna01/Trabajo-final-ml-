# Proyecto Final - Machine Learning

## Clasificación y Clustering con Aplicación Web

Este proyecto integra técnicas de aprendizaje supervisado y no supervisado utilizando Python y una aplicación web construida con Streamlit. Incluye dos notebooks de análisis, un script para entrenamiento en producción y una interfaz para realizar predicciones en tiempo real.

---

## Autores

- **Johan Stiven Sinisterra Campaz**
- **Kevin Fernando Serna Goyes**
- **Juan David Quintero Pimentel**

---

## Contenidos del Proyecto

### 1. Predicción de Churn (Telco) - Clasificación

Notebook enfocado en predecir si un cliente de telecomunicaciones abandonará el servicio.

#### Pasos principales

**Limpieza de datos**
- Conversión de TotalCharges a formato numérico
- Eliminación de valores nulos
- Eliminación de la columna customerID
- Mapeo de Churn a valores 0 y 1

**Preprocesamiento**
- One-Hot Encoding para variables categóricas
- MinMaxScaler para normalización
- División en Train/Test (80/20)

**Modelado**
- Regresión Logística
- KNN (k = 9)

**Evaluación**
- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de confusión
- Curva ROC

**Resultado:** La Regresión Logística obtiene el mejor AUC aproximado de **0.84**.

---

### 2. Clustering de Clientes (Credit Cards) - K-Means

Notebook enfocado en segmentar clientes de tarjetas de crédito.

#### Pasos principales

**Limpieza**
- Relleno de valores nulos mediante la mediana
- Eliminación de la columna CUST_ID

**Escalado**
- StandardScaler para estandarización

**Selección de K**
- Método del codo → K = 4

**Modelado**
- K-Means con n_clusters = 4

**Análisis de perfiles**
- Silhouette score aproximadamente **0.45**

**Perfiles identificados:**
1. **Bajo uso / Ahorradores**
2. **Clientes VIP / Gastadores**
3. **Usuarios de efectivo**
4. **Deudores con alto balance**

---

## Script de Entrenamiento - `entrenar_modelos.py`

Este script automatiza el preprocesamiento y guardado de los modelos entrenados, así como los escaladores y columnas necesarias.

### Archivos generados

**Modelos (carpeta `models/`):**
- `modelo_logistica.pkl` - Regresión Logística
- `modelo_knn.pkl` - K-Nearest Neighbors
- `modelo_kmeans.pkl` - K-Means Clustering

**Escaladores (carpeta `scalers/`):**
- `scaler_telco.pkl` - MinMaxScaler para Telco
- `scaler_cc.pkl` - StandardScaler para Credit Card

**Metadatos (carpeta `data/`):**
- `columnas_telco.pkl` - Nombres de columnas procesadas (Telco)
- `columnas_cc.pkl` - Nombres de columnas procesadas (Credit Card)

---

## Aplicación Web - `app.py`

Aplicación construida en Streamlit que permite:

### Clasificación (Churn)
- Ingresar datos del cliente mediante formulario interactivo
- Aplicar preprocesamiento automático
- Obtener probabilidad de abandono usando Regresión Logística o KNN
- Visualizar recomendaciones según nivel de riesgo

### Clustering (Credit Cards)
- Ingresar datos financieros del cliente
- Aplicar escalado y alineación de columnas
- Asignación a uno de los 4 clusters
- Descripción del perfil correspondiente
- Estrategias de marketing sugeridas

---

## Estructura del Proyecto

```
Trabajo-final-ml-/
│
├── app.py                          # Aplicación web Streamlit
├── entrenar_modelos.py             # Script de entrenamiento
├── requirements.txt                # Dependencias Python
├── .gitignore                      # Archivos ignorados por Git
│
├── models/                         # Modelos entrenados
│   ├── modelo_logistica.pkl
│   ├── modelo_knn.pkl
│   └── modelo_kmeans.pkl
│
├── scalers/                        # Escaladores/Normalizadores
│   ├── scaler_telco.pkl
│   └── scaler_cc.pkl
│
├── data/                           # Metadatos de columnas
│   ├── columnas_telco.pkl
│   └── columnas_cc.pkl
│
├── datasets/                       # Datasets originales
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── CC GENERAL.csv
│
└── notebooks/                      # Análisis exploratorio
    ├── Proyecto_Final_Telco.ipynb
    └── Proyecto_Final_Clustering_Tarjetas.ipynb
```

---

## Tecnologías Utilizadas

- **Python 3.13**
- **Scikit-learn** - Algoritmos de Machine Learning
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Streamlit** - Framework para aplicación web
- **Matplotlib** - Visualización de datos
- **Seaborn** - Visualización estadística
- **Pickle** - Serialización de modelos

---

## Cómo Ejecutarlo

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar los modelos

```bash
python entrenar_modelos.py
```

Este comando:
- Carga los datasets desde `datasets/`
- Preprocesa los datos
- Entrena los 3 modelos
- Guarda modelos en `models/`
- Guarda escaladores en `scalers/`
- Guarda metadatos en `data/`

### 3. Iniciar la aplicación

```bash
streamlit run app.py
```

### 4. Abrir en el navegador

La aplicación se abrirá automáticamente en:

```
http://localhost:8501
```

