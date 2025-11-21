# 🤖 Proyecto Final - Machine Learning Supervisado y No Supervisado

Proyecto académico de Machine Learning que implementa modelos supervisados (clasificación) y no supervisados (clustering) con aplicación web interactiva.

## 📋 Descripción

Este proyecto consta de dos análisis principales:

### 1. **Predicción de Churn (Telco Customer Churn)**
- **Objetivo:** Predecir si un cliente de telecomunicaciones cancelará su servicio.
- **Modelos:** Regresión Logística y K-Nearest Neighbors (KNN)
- **Métricas:** ROC Curve, AUC, Matriz de Confusión, Accuracy, Precision, Recall, F1-Score

### 2. **Segmentación de Clientes (Credit Card Dataset)**
- **Objetivo:** Agrupar clientes de tarjetas de crédito en perfiles de comportamiento.
- **Modelo:** K-Means Clustering
- **Análisis:** Método del Codo, Silhouette Score, Interpretación de Perfiles

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/nombre-repo.git
cd nombre-repo
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
```

### 3. Activar entorno virtual

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 📊 Notebooks (Análisis Exploratorio)

### Modelo Supervisado - Churn
```bash
jupyter notebook Proyecto_Final_Telco.ipynb
```

### Modelo No Supervisado - Clustering
```bash
jupyter notebook Proyecto_Final_Clustering_Tarjetas.ipynb
```

## 🌐 Aplicación Web

### 1. Entrenar y exportar modelos
```bash
python entrenar_modelos.py
```

Este script:
- Carga y preprocesa los datasets
- Entrena Regresión Logística, KNN y K-Means
- Exporta los modelos como archivos `.pkl`

### 2. Ejecutar la aplicación web
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎯 Funcionalidades de la Web

### 📊 Regresión Logística (Churn)
- Formulario con variables del cliente
- Predicción de probabilidad de abandono (%)
- Clasificación: Yes/No

### 🔍 K-Nearest Neighbors (Churn)
- Mismo formulario que Regresión Logística
- Clasificación basada en vecinos cercanos
- Resultado: Yes/No

### 💳 K-Means (Clustering)
- Formulario con features numéricas del cliente
- Asignación a cluster (0-3)
- Descripción del perfil del cluster:
  - **Grupo 0:** Ahorradores / Bajo Uso
  - **Grupo 1:** Gastadores VIP
  - **Grupo 2:** Usuarios de Efectivo
  - **Grupo 3:** Alto Balance / Deudores

## 📁 Estructura del Proyecto

```
├── app.py                                    # Aplicación web Streamlit
├── entrenar_modelos.py                       # Script de entrenamiento
├── Proyecto_Final_Telco.ipynb               # Notebook Churn (supervisado)
├── Proyecto_Final_Clustering_Tarjetas.ipynb # Notebook Clustering (no supervisado)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset Telco
├── CC GENERAL.csv                            # Dataset Credit Card
├── modelo_logistica.pkl                      # Modelo exportado
├── modelo_knn.pkl                            # Modelo exportado
├── modelo_kmeans.pkl                         # Modelo exportado
├── scaler_telco.pkl                          # Scaler para Telco
├── scaler_cc.pkl                             # Scaler para Credit Card
├── columnas_telco.pkl                        # Columnas procesadas
├── columnas_cc.pkl                           # Columnas procesadas
├── requirements.txt                          # Dependencias
├── .gitignore                                # Archivos ignorados
└── README.md                                 # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.13**
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Machine Learning
- **Matplotlib & Seaborn** - Visualización
- **Streamlit** - Aplicación web interactiva

## 📈 Resultados

### Modelo Supervisado (Churn)
- **Regresión Logística:** AUC ~0.84
- **KNN:** AUC ~0.82
- Ambos modelos muestran buen rendimiento en la detección de clientes en riesgo

### Modelo No Supervisado (Clustering)
- **K óptimo:** 4 clusters
- **Silhouette Score:** ~0.45
- Perfiles claramente diferenciados por comportamiento financiero

## 👥 Autor

Kevin Serna - Proyecto Final Machine Learning

## 📝 Licencia

Este proyecto es de uso académico.

## 🙏 Agradecimientos

- Dataset Telco: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Dataset Credit Card: [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

