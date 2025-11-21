"""
Script para entrenar y exportar los modelos de Machine Learning
Genera archivos .pkl que serán usados por la aplicación web
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Iniciando entrenamiento de modelos...")

# ============================================
# PARTE 1: MODELOS SUPERVISADOS (TELCO CHURN)
# ============================================
print("\n1. Cargando dataset Telco...")
df_telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Limpieza
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
df_telco.dropna(inplace=True)
df_telco.drop('customerID', axis=1, inplace=True)
df_telco['Churn'] = df_telco['Churn'].map({'Yes': 1, 'No': 0})

# Preprocesamiento
X = df_telco.drop('Churn', axis=1)
y = df_telco['Churn']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler_telco = MinMaxScaler()
X_encoded[numerical_cols] = scaler_telco.fit_transform(X_encoded[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar Regresión Logística
print("   -> Entrenando Regresion Logistica...")
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Entrenar KNN
print("   -> Entrenando KNN...")
knn_model = KNeighborsClassifier(n_neighbors=9)
knn_model.fit(X_train, y_train)

# Guardar modelos supervisados
with open('modelo_logistica.pkl', 'wb') as f:
    pickle.dump(log_model, f)

with open('modelo_knn.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('scaler_telco.pkl', 'wb') as f:
    pickle.dump(scaler_telco, f)

# Guardar nombres de columnas (crítico para la web)
columnas_telco = X_encoded.columns.tolist()
with open('columnas_telco.pkl', 'wb') as f:
    pickle.dump(columnas_telco, f)

print("   -> Modelos Telco guardados: modelo_logistica.pkl, modelo_knn.pkl, scaler_telco.pkl")

# ============================================
# PARTE 2: MODELO NO SUPERVISADO (CLUSTERING)
# ============================================
print("\n2. Cargando dataset Credit Card...")
df_cc = pd.read_csv('CC GENERAL.csv')

# Limpieza
df_cc['MINIMUM_PAYMENTS'] = df_cc['MINIMUM_PAYMENTS'].fillna(df_cc['MINIMUM_PAYMENTS'].median())
df_cc['CREDIT_LIMIT'] = df_cc['CREDIT_LIMIT'].fillna(df_cc['CREDIT_LIMIT'].median())
if 'CUST_ID' in df_cc.columns:
    df_cc.drop('CUST_ID', axis=1, inplace=True)

# Escalado
scaler_cc = StandardScaler()
X_scaled_cc = scaler_cc.fit_transform(df_cc)

# Entrenar K-Means
print("   -> Entrenando K-Means (k=4)...")
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(X_scaled_cc)

# Guardar modelo clustering
with open('modelo_kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)

with open('scaler_cc.pkl', 'wb') as f:
    pickle.dump(scaler_cc, f)

columnas_cc = df_cc.columns.tolist()
with open('columnas_cc.pkl', 'wb') as f:
    pickle.dump(columnas_cc, f)

print("   -> Modelo K-Means guardado: modelo_kmeans.pkl, scaler_cc.pkl")

print("\nTodos los modelos entrenados y exportados exitosamente!")
print("Ahora puedes ejecutar la aplicación web con: streamlit run app.py")

