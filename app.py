import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import preprocesamiento, modelado, metricas

st.title("APP Modelos de Machine Learning")
st.write("Esta aplicación te llevará paso a paso para entrenar un modelo de Machine Learning.")

# Inicializar estados
if 'datos_cargados' not in st.session_state:
    st.session_state['datos_cargados'] = False
if 'target' not in st.session_state:
    st.session_state['target'] = False
if 'preprocesado' not in st.session_state:
    st.session_state['preprocesado'] = False
if 'modelo_entrenado' not in st.session_state:
    st.session_state['modelo_entrenado'] = False
if 'nulos_procesados' not in st.session_state:
    st.session_state['nulos_procesados'] = False

# 1. Carga de datos
st.header("1. Cargar Datos")
st.write("Sube un archivo CSV con tus datos para comenzar. Asegúrate que tenga una columna con la variable objetivo")
file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    # Vista previa de datos
    st.write("Vista previa de los datos:", data.head())

    # Guardar datos en el estado de sesión
    st.session_state['data'] = data
    st.session_state['datos_cargados'] = True

# 2. Definir target y identificador (si existe)
if st.session_state['datos_cargados']:
    st.header("2. Definir variable objetivo y ID (opcional)")
    st.write("Selecciona la variable objetivo y convierte las variables categóricas a numéricas (con One-hot encoding).")
    
    # Preguntar por la columna de ID
    id_col = st.selectbox("Selecciona la columna de ID (opcional):",
                          [None] + list(st.session_state['data'].columns))
    
    # Guardar en session state
    st.session_state['id_col'] = id_col
    
    # Seleccionar variable objetivo
    target = st.selectbox("Selecciona tu variable objetivo:", 
                          st.session_state['data'].columns)
    # Revisar si variable target es numerica. No se puede entrenar un modelo con una variable objetivo no numérica
    if st.session_state['data'][target].dtype not in ['int64', 'float64']:
        st.warning("La variable objetivo seleccionada no es numérica")
        st.stop()

    st.session_state['target'] = target

# 3. Procesamiento de datos
if st.session_state['target']:
    st.header("3. Preprocesamiento de Datos")
    st.write("Manejo de valores nulos y codificación de variables categóricas.")

    # 3.1 Revisar valores nulos (siempre disponible)
    total_columnas, columnas_con_nulos = preprocesamiento.contar_columnas_y_nulos(st.session_state['data'])
    
    # Mostrar estado actual de nulos
    if columnas_con_nulos > 0:
        st.warning(f"Estado actual: **{total_columnas}** columnas totales, **{columnas_con_nulos}** con valores nulos")
    else:
        st.success(f"Estado actual: **{total_columnas}** columnas totales, **sin valores nulos**")
    
    # Mostrar opciones de procesamiento solo si hay nulos
    if columnas_con_nulos > 0:
        # Selector de acción
        opcion_nulos = st.selectbox(
            "¿Cómo quieres manejar los valores nulos? En el caso de sustituirlos, solo podemos hacerlo para variables numéricas", 
            ("Eliminar filas con nulos", "Reemplazar nulos por la media")
        )

        # Botón para procesar la acción elegida
        if st.button("Procesar valores nulos"):
            if opcion_nulos == "Eliminar filas con nulos":
                st.session_state['data'] = preprocesamiento.eliminar_nulos(st.session_state['data'])
                st.success("Filas con nulos eliminadas")
                st.session_state['nulos_procesados'] = True
            else:
                st.session_state['data'] = preprocesamiento.reemplazar_nulos_media(st.session_state['data'])
                st.success("Nulos reemplazados por la media de las columnas")
                st.session_state['nulos_procesados'] = True
            #st.rerun()  # Actualizar la página para reflejar los cambios
    
    # One-hot encoding de variables categóricas
    if st.session_state['nulos_procesados']:
        # Codificar variables categóricas
        if st.button("Aplicar One-hot encoding a variables categóricas"):
            # Separar ID si existe
            id_col = st.session_state.get('id_col', None)
            data_encoded = st.session_state['data'].copy()

            if id_col:
                data_encoded = data_encoded.drop(columns=[id_col])
            
            # Aplicar codificación
            data_encoded = preprocesamiento.codificar_categoricas(data_encoded)
            
            # Separar X e y
            target = st.session_state['target']
            y = data_encoded[target]
            X = data_encoded.drop(columns=[target])
            
            # Guardar en session state
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['preprocesado'] = True
            
            st.success("Variables categóricas codificadas. Listo para entrenar el modelo.")
            st.write(f"Dimensiones finales: {X.shape[0]} filas y {X.shape[1]} características")
            
        # Si ya está preprocesado, mostrar información
        if st.session_state['preprocesado']:
            st.info("Datos preprocesados y listos para el modelado.")
            if 'X' in st.session_state and 'y' in st.session_state:
                st.write(f"Variables predictoras (X): {st.session_state['X'].shape}")
                st.write(f"Variable objetivo (y): {st.session_state['y'].shape}")



# 4. Análisis exploratorio
if st.session_state['preprocesado']:
    st.header("4. Análisis Exploratorio")

    if st.button("Mostrar correlación"):
        corr = st.session_state['X'].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# 5. Entrenamiento de modelo
if st.session_state['preprocesado']:
    st.header("5. Entrenamiento del Modelo")

    test_size = st.slider("Proporción de test (%)", 10, 50, 30)

    if st.button("Entrenar modelo"):
        X = st.session_state['X']
        y = st.session_state['y']

        modelo, X_test, y_test = modelado.entrenar_modelo_logistico(X, y, test_size=test_size/100)
        st.session_state['modelo'] = modelo
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['modelo_entrenado'] = True

        st.success("Modelo entrenado ✅")

# 6. Métricas
if st.session_state['modelo_entrenado']:
    st.header("6. Métricas del Modelo")

    modelo = st.session_state['modelo']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    precision = metricas.obtener_precision(modelo, X_test, y_test)
    st.write(f"🔍 Precisión: **{precision:.2%}**")

    if st.button("Mostrar matriz de confusión"):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)
