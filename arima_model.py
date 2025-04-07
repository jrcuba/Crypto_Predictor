import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def predict(data, steps=20):
    """
    Predice el precio futuro utilizando un modelo ARIMA.
    :param data: Serie de tiempo (lista o array) de precios.
    :param steps: Número de períodos a predecir.
    :return: Array con la predicción (de largo 'steps').
    """
    series = pd.Series(data)
    # Se ajusta un ARIMA(1,1,1); este orden puede optimizarse
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.to_numpy()
