import numpy as np
from sklearn.linear_model import LinearRegression

def predict(data, steps=20):
    """
    Predice el precio futuro utilizando regresión lineal.
    :param data: Serie de tiempo (lista o array) de precios.
    :param steps: Número de períodos a predecir.
    :return: Array con la predicción.
    """
    data = np.array(data)
    X = np.arange(len(data)).reshape(-1, 1)
    y = data
    model = LinearRegression()
    model.fit(X, y)
    X_pred = np.arange(len(data), len(data) + steps).reshape(-1, 1)
    forecast = model.predict(X_pred)
    # Ajustar el pronóstico para que comience cerca del último valor de datos
    forecast += data[-1] - forecast[0]
    return forecast
