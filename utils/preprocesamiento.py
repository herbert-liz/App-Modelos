import pandas as pd

# Funciones de preprocesamiento de datos

# Cuenta cuantas columnas tiene valores nulos
def contar_columnas_y_nulos(df):
    total_columnas = df.shape[1]
    columnas_con_nulos = df.isnull().any().sum()
    
    return total_columnas, columnas_con_nulos

# Eliminar filas con valores nulos
def eliminar_nulos(df):
    return df.dropna()

# Cambiar nulos por la media de la columna (solo para columnas numéricas)
def reemplazar_nulos_media(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    return df.fillna(df[num_cols].mean())

# Codificar variables categóricas usando One-hot encoding
def codificar_categoricas(df):
    return pd.get_dummies(df)
