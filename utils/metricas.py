from sklearn.metrics import accuracy_score

def obtener_precision(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return accuracy_score(y_test, y_pred)
