# Proyecto Final - Machine Learning Supervisado y No Supervisado

Proyecto académico de Machine Learning que implementa modelos supervisados (clasificación) y no supervisados (clustering) con aplicación web interactiva.

## Descripción

Este proyecto consta de dos análisis principales de Machine Learning aplicados a casos de uso empresariales:

### 1. Predicción de Churn (Telco Customer Churn)

**Objetivo:** Predecir si un cliente de telecomunicaciones cancelará su servicio.

**Modelos implementados:**
- Regresión Logística
- K-Nearest Neighbors (KNN)

**Métricas de evaluación:**
- ROC Curve
- AUC (Area Under the Curve)
- Matriz de Confusión
- Accuracy, Precision, Recall, F1-Score

### 2. Segmentación de Clientes (Credit Card Dataset)

**Objetivo:** Agrupar clientes de tarjetas de crédito en perfiles de comportamiento homogéneos.

**Modelo implementado:**
- K-Means Clustering

**Análisis realizado:**
- Método del Codo (Elbow Method)
- Silhouette Score
- Interpretación de Perfiles de Clusters

## Instalación

### Requisitos previos
- Python 3.13 o superior
- pip (gestor de paquetes de Python)

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/nombre-repo.git
cd nombre-repo
```

### 2. Crear entorno virtual (recomendado)

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

## Uso del Proyecto

### Análisis Exploratorio (Notebooks Jupyter)

Para revisar el análisis exploratorio de datos y el proceso de desarrollo de los modelos:

**Modelo Supervisado - Predicción de Churn:**
```bash
jupyter notebook Proyecto_Final_Telco.ipynb
```

**Modelo No Supervisado - Clustering:**
```bash
jupyter notebook Proyecto_Final_Clustering_Tarjetas.ipynb
```

### Aplicación Web Interactiva

#### Paso 1: Entrenar y exportar modelos

Antes de ejecutar la aplicación web, es necesario entrenar los modelos:

```bash
python entrenar_modelos.py
```

Este script realiza las siguientes operaciones:
- Carga y preprocesa los datasets
- Entrena Regresión Logística, KNN y K-Means
- Exporta los modelos como archivos `.pkl`
- Guarda los scalers y nombres de columnas necesarios

#### Paso 2: Ejecutar la aplicación web

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Funcionalidades de la Aplicación Web

### Regresión Logística (Predicción de Churn)

- Formulario interactivo con variables del cliente
- Predicción de probabilidad de abandono (porcentaje)
- Clasificación binaria: Yes/No
- Recomendaciones según nivel de riesgo

**Variables de entrada:**
- Datos demográficos (género, edad, dependientes)
- Servicios contratados (internet, teléfono, streaming)
- Información de contrato (tipo, método de pago)
- Datos financieros (antigüedad, cargos mensuales)

### K-Nearest Neighbors (Predicción de Churn)

- Mismo formulario que Regresión Logística
- Clasificación basada en vecinos cercanos (k=9)
- Resultado: Yes/No con interpretación

### K-Means Clustering (Segmentación)

- Formulario con features numéricas del cliente
- Asignación automática a cluster (0-3)
- Descripción detallada del perfil del cluster
- Estrategias de marketing sugeridas

**Perfiles de clusters identificados:**
- **Grupo 0:** Ahorradores / Bajo Uso
- **Grupo 1:** Gastadores VIP
- **Grupo 2:** Usuarios de Efectivo
- **Grupo 3:** Alto Balance / Deudores

## Estructura del Proyecto

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
├── requirements.txt                          # Dependencias Python
├── .gitignore                                # Archivos ignorados
└── README.md                                 # Este archivo
```

## Tecnologías Utilizadas

### Lenguaje y Framework
- **Python 3.13** - Lenguaje de programación

### Bibliotecas de Data Science
- **Pandas 2.3.3** - Manipulación y análisis de datos
- **NumPy 2.3.5** - Operaciones numéricas y matrices
- **Matplotlib 3.10.7** - Visualización de datos (gráficos)
- **Seaborn 0.13.2** - Visualización estadística avanzada

### Machine Learning
- **Scikit-learn 1.7.2** - Algoritmos de ML y preprocesamiento
  - Regresión Logística
  - K-Nearest Neighbors
  - K-Means Clustering
  - MinMaxScaler y StandardScaler
  - Métricas de evaluación

### Aplicación Web
- **Streamlit 1.51.0** - Framework para aplicaciones web interactivas

## Resultados del Proyecto

### Modelos Supervisados (Predicción de Churn)

**Regresión Logística:**
- AUC-ROC: ~0.84
- Proporciona probabilidades interpretables
- Rápido en inferencia

**K-Nearest Neighbors:**
- AUC-ROC: ~0.82
- Efectivo para patrones no lineales
- k=9 vecinos

**Conclusión:** Ambos modelos muestran buen rendimiento en la detección de clientes en riesgo de abandono, siendo la Regresión Logística ligeramente superior en términos de AUC.

### Modelo No Supervisado (Segmentación de Clientes)

**K-Means Clustering:**
- Número óptimo de clusters: 4
- Silhouette Score: ~0.45
- Perfiles claramente diferenciados por comportamiento financiero

**Insights obtenidos:**
- Identificación de 4 segmentos distintos de clientes
- Cada cluster tiene características financieras únicas
- Permite personalización de estrategias de marketing

## Metodología

### Pipeline de Modelos Supervisados

1. **Carga de datos:** Dataset Telco Customer Churn
2. **Limpieza:** Manejo de valores nulos y conversión de tipos
3. **Feature Engineering:** One-Hot Encoding de variables categóricas
4. **Normalización:** MinMaxScaler para variables numéricas
5. **División:** 80% entrenamiento, 20% prueba (stratified)
6. **Entrenamiento:** Regresión Logística y KNN
7. **Evaluación:** Métricas de clasificación y ROC-AUC

### Pipeline de Modelo No Supervisado

1. **Carga de datos:** Dataset Credit Card
2. **Limpieza:** Imputación de valores faltantes con mediana
3. **Normalización:** StandardScaler (estandarización)
4. **Determinación de k:** Método del Codo y Silhouette
5. **Entrenamiento:** K-Means con k=4
6. **Evaluación:** Análisis de clusters e interpretación

## Consideraciones Técnicas

### Preprocesamiento

- **One-Hot Encoding:** Transformación de variables categóricas a formato binario
- **Normalización:** Escalado de variables numéricas para evitar dominancia por escala
- **Imputación:** Uso de mediana para valores faltantes (robusto ante outliers)

### Validación

- **Stratified Split:** Mantiene proporción de clases en train/test
- **Random State:** Reproducibilidad de resultados (seed=42)
- **Cross-Validation:** Implementado en notebooks para validación robusta

## Autor

**Kevin Serna**  
Proyecto Final - Machine Learning  
Estudiante de Ingeniería

## Fuentes de Datos

- **Dataset Telco Customer Churn:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Dataset Credit Card:** [Kaggle - Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

## Licencia

Este proyecto es de uso académico y educativo.

## Notas Adicionales

### Reentrenamiento de Modelos

Si desea reentrenar los modelos con diferentes hiperparámetros:

1. Modificar `entrenar_modelos.py` con los nuevos parámetros
2. Ejecutar: `python entrenar_modelos.py`
3. Los nuevos modelos sobrescribirán los archivos `.pkl` existentes

### Personalización de la Aplicación

El archivo `app.py` puede ser personalizado para:
- Agregar nuevas métricas de visualización
- Modificar umbrales de decisión
- Incluir nuevos modelos
- Personalizar la interfaz de usuario

### Troubleshooting

**Error: No module named 'X'**
- Solución: Verificar que todas las dependencias estén instaladas con `pip install -r requirements.txt`

**Error: No such file or directory: 'modelo_X.pkl'**
- Solución: Ejecutar primero `python entrenar_modelos.py` para generar los modelos

**Error: Mismatch in number of features**
- Solución: Asegurarse de que los datos de entrada tengan el mismo formato que los datos de entrenamiento
