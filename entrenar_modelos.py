"""
Script de Entrenamiento de Modelos de Machine Learning

Este script realiza el entrenamiento de los modelos de clasificación
y clustering, y exporta los modelos entrenados junto con los recursos
necesarios para su uso.

Modelos implementados:
1. Regresión Logística (Clasificación de Churn)
2. K-Nearest Neighbors (Clasificación de Churn)
3. K-Means (Clustering de clientes)

Autores: Kevin Serna, Johan Stiven Sinisterra , Juan David Quintero
Proyecto Final - Machine Learning
"""

import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def entrenar_modelos_telco():
    """
    Entrenamiento de los modelos de clasificación para predicción de Churn.
    
    1. Carga y limpieza de datos
    2. Preprocesamiento (One-Hot Encoding y escalado)
    3. División train/test
    4. Entrenamiento de Regresión Logística y KNN
    5. Exportación de modelos y recursos
    
    Retorna un tupla con los modelos entrenados, scaler y nombres de columnas:

        tuple: Modelos entrenados, scaler y nombres de columnas
    """
    print("\n" + "="*60)
    print("FASE 1: ENTRENAMIENTO DE MODELOS SUPERVISADOS (TELCO CHURN)")
    print("="*60)
    
    # Carga de datos
    print("\n1. Cargando dataset Telco Customer Churn...")
    df_telco = pd.read_csv('datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"   - Dimensiones originales: {df_telco.shape}")
    
    # Limpieza de datos
    print("\n2. Realizando limpieza de datos...")
    
    # Convertir TotalCharges a numérico (manejo de errores)
    df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
    nulos_antes = df_telco.shape[0]
    
    # Eliminar filas con valores nulos
    df_telco.dropna(inplace=True)
    nulos_despues = df_telco.shape[0]
    print(f"   - Registros eliminados por valores nulos: {nulos_antes - nulos_despues}")
    
    # Eliminar columna de ID (no aporta información predictiva)
    df_telco.drop('customerID', axis=1, inplace=True)
    print(f"   - Columna 'customerID' eliminada")
    
    # Convertir variable target a numérica
    df_telco['Churn'] = df_telco['Churn'].map({'Yes': 1, 'No': 0})
    print(f"   - Variable target convertida a formato numérico")
    
    # Separación de features y target
    print("\n3. Preparando features y target...")
    X = df_telco.drop('Churn', axis=1)
    y = df_telco['Churn']
    
    # Identificar tipos de columnas
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"   - Variables categóricas: {len(categorical_cols)}")
    print(f"   - Variables numéricas: {len(numerical_cols)}")
    
    # One-Hot Encoding
    # drop_first=True para evitar la trampa de las variables dummy
    print("\n4. Aplicando One-Hot Encoding...")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"   - Dimensiones después de encoding: {X_encoded.shape}")
    
    # Escalado de variables numéricas
    # MinMaxScaler normaliza los valores al rango [0, 1]
    print("\n5. Aplicando normalización MinMax a variables numéricas...")
    scaler_telco = MinMaxScaler()
    X_encoded[numerical_cols] = scaler_telco.fit_transform(X_encoded[numerical_cols])
    print(f"   - Variables escaladas al rango [0, 1]")
    
    # División de datos
    # stratify=y mantiene la proporción de clases en train y test
    print("\n6. Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"   - Conjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"   - Conjunto de prueba: {X_test.shape[0]} registros")
    
    # Entrenamiento de Regresión Logística
    print("\n7. Entrenando modelo de Regresión Logística...")
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    
    # Calcular accuracy en conjunto de prueba
    accuracy_log = log_model.score(X_test, y_test)
    print(f"   - Modelo entrenado exitosamente")
    print(f"   - Accuracy en test: {accuracy_log:.4f}")
    
    # Entrenamiento de KNN
    print("\n8. Entrenando modelo K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=9)
    knn_model.fit(X_train, y_train)
    
    # Calcular accuracy en conjunto de prueba
    accuracy_knn = knn_model.score(X_test, y_test)
    print(f"   - Modelo entrenado exitosamente")
    print(f"   - Accuracy en test: {accuracy_knn:.4f}")
    print(f"   - Número de vecinos (k): 9")
    
    # Exportación de modelos
    print("\n9. Exportando modelos y recursos...")
    
    with open('models/modelo_logistica.pkl', 'wb') as f:
        pickle.dump(log_model, f)
    print(f"   - models/modelo_logistica.pkl guardado")
    
    with open('models/modelo_knn.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print(f"   - models/modelo_knn.pkl guardado")
    
    with open('scalers/scaler_telco.pkl', 'wb') as f:
        pickle.dump(scaler_telco, f)
    print(f"   - scalers/scaler_telco.pkl guardado")
    
    # Guardar nombres de columnas (crítico para predicción)
    columnas_telco = X_encoded.columns.tolist()
    with open('data/columnas_telco.pkl', 'wb') as f:
        pickle.dump(columnas_telco, f)
    print(f"   - data/columnas_telco.pkl guardado ({len(columnas_telco)} columnas)")
    
    print("\n" + "="*60)
    print("FASE 1 COMPLETADA: Modelos supervisados entrenados")
    print("="*60)
    
    return log_model, knn_model, scaler_telco, columnas_telco


def entrenar_modelo_clustering():
    """
    Entrenamiento del modelo de clustering K-Means para segmentación de clientes.
    
    1. Carga y limpieza de datos
    2. Imputación de valores faltantes
    3. Escalado de características
    4. Entrenamiento de K-Means
    5. Exportación de modelo y recursos
    
    Retorna un tupla con el modelo entrenado, scaler y nombres de columnas:

        tuple: Modelo entrenado, scaler y nombres de columnas
    """
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO DE MODELO NO SUPERVISADO (CLUSTERING)")
    print("="*60)
    
    # Carga de datos
    print("\n1. Cargando dataset Credit Card...")
    df_cc = pd.read_csv('datasets/CC GENERAL.csv')
    print(f"   - Dimensiones originales: {df_cc.shape}")
    
    # Análisis de valores faltantes
    print("\n2. Analizando valores faltantes...")
    nulos_por_columna = df_cc.isnull().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    
    if len(columnas_con_nulos) > 0:
        print(f"   - Columnas con valores nulos:")
        for col, count in columnas_con_nulos.items():
            print(f"     * {col}: {count} valores nulos")
    
    # Imputación de valores faltantes con la mediana
    # La mediana es robusta ante valores atípicos
    print("\n3. Imputando valores faltantes con la mediana...")
    df_cc['MINIMUM_PAYMENTS'] = df_cc['MINIMUM_PAYMENTS'].fillna(
        df_cc['MINIMUM_PAYMENTS'].median()
    )
    df_cc['CREDIT_LIMIT'] = df_cc['CREDIT_LIMIT'].fillna(
        df_cc['CREDIT_LIMIT'].median()
    )
    print(f"   - Valores faltantes imputados")
    
    # Eliminar columna de ID
    if 'CUST_ID' in df_cc.columns:
        df_cc.drop('CUST_ID', axis=1, inplace=True)
        print(f"   - Columna 'CUST_ID' eliminada")
    
    # Escalado de datos
    # StandardScaler estandariza: media=0, desviación estándar=1
    print("\n4. Aplicando estandarización (StandardScaler)...")
    scaler_cc = StandardScaler()
    X_scaled_cc = scaler_cc.fit_transform(df_cc)
    print(f"   - Datos estandarizados (media=0, std=1)")
    
    # Entrenamiento de K-Means
    # n_init=10 ejecuta el algoritmo 10 veces con diferentes centroides iniciales
    print("\n5. Entrenando modelo K-Means...")
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled_cc)
    
    # Calcular inercia (suma de distancias cuadradas al centroide más cercano)
    inertia = kmeans_model.inertia_
    print(f"   - Modelo entrenado exitosamente")
    print(f"   - Número de clusters: 4")
    print(f"   - Inercia: {inertia:.2f}")
    
    # Análisis de distribución de clusters
    labels = kmeans_model.labels_
    print(f"\n6. Distribución de clientes por cluster:")
    for i in range(4):
        count = (labels == i).sum()
        percentage = (count / len(labels)) * 100
        print(f"   - Cluster {i}: {count} clientes ({percentage:.2f}%)")
    
    # Exportación de modelo y recursos
    print("\n7. Exportando modelo y recursos...")
    
    with open('models/modelo_kmeans.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    print(f"   - models/modelo_kmeans.pkl guardado")
    
    with open('scalers/scaler_cc.pkl', 'wb') as f:
        pickle.dump(scaler_cc, f)
    print(f"   - scalers/scaler_cc.pkl guardado")
    
    # Guardar nombres de columnas
    columnas_cc = df_cc.columns.tolist()
    with open('data/columnas_cc.pkl', 'wb') as f:
        pickle.dump(columnas_cc, f)
    print(f"   - data/columnas_cc.pkl guardado ({len(columnas_cc)} columnas)")
    
    print("\n" + "="*60)
    print("FASE 2 COMPLETADA: Modelo de clustering entrenado")
    print("="*60)
    
    return kmeans_model, scaler_cc, columnas_cc


def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    """
    print("\n" + "#"*60)
    print("# SISTEMA DE ENTRENAMIENTO DE MODELOS DE MACHINE LEARNING")
    print("#"*60)
    print("\nIniciando proceso de entrenamiento...")
    
    # Fase 1: Modelos supervisados
    log_model, knn_model, scaler_telco, columnas_telco = entrenar_modelos_telco()
    
    # Fase 2: Modelo no supervisado
    kmeans_model, scaler_cc, columnas_cc = entrenar_modelo_clustering()
    
    # Resumen final
    print("\n" + "#"*60)
    print("# RESUMEN FINAL")
    print("#"*60)
    print("\nModelos entrenados y exportados exitosamente:")
    print("\nModelos Supervisados (Churn Prediction):")
    print("  1. models/modelo_logistica.pkl - Regresión Logística")
    print("  2. models/modelo_knn.pkl - K-Nearest Neighbors")
    print("  3. scalers/scaler_telco.pkl - MinMax Scaler")
    print("  4. data/columnas_telco.pkl - Nombres de columnas")
    
    print("\nModelo No Supervisado (Clustering):")
    print("  5. models/modelo_kmeans.pkl - K-Means Clustering")
    print("  6. scalers/scaler_cc.pkl - Standard Scaler")
    print("  7. data/columnas_cc.pkl - Nombres de columnas")
    
    print("\n" + "#"*60)
    print("# SIGUIENTE PASO")
    print("#"*60)
    print("\nPara ejecutar la aplicación web, utilice el comando:")
    print("  streamlit run app.py")
    print("\n" + "#"*60)


if __name__ == "__main__":
    main()
