import numpy as np

def predict(data, steps=20):
    """
    Predice el precio futuro usando la solución analítica de la EDO: dP/dt = a * P + b.
    Se ajustan los parámetros a partir de diferencias finitas.
    :param data: Serie de tiempo (lista o array) de precios.
    :param steps: Número de períodos a predecir.
    :return: Array con la predicción.
    """
    data = np.array(data)
    y_diff = np.diff(data)
    A = np.vstack([data[:-1], np.ones(len(data)-1)]).T
    params, _, _, _ = np.linalg.lstsq(A, y_diff, rcond=None)
    a, b = params
    P0 = data[-1]
    t = np.linspace(0, steps, steps+1)
    if abs(a) < 1e-8:
        forecast = P0 + b * t
    else:
        forecast = (P0 + b / a) * np.exp(a * t) - b / a
    return forecast[1:]  # Se excluye el valor inicial
