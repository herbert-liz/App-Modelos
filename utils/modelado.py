from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def entrenar_modelo_logistico(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    return modelo, X_test, y_test
