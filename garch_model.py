# garch_model.py
import numpy as np
import pandas as pd
from arch import arch_model

def predict(data, steps=20):
    """
    Predice el precio futuro utilizando un modelo GARCH(1,1).
    :param data: Serie de tiempo (lista o array) de precios.
    :param steps: Número de períodos a predecir.
    :return: Array con la predicción.
    """
    series = pd.Series(data)
    returns = series.pct_change().dropna() * 100  # retornos en porcentajes
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=steps)
    # Extraemos la media pronosticada para el último período de la predicción
    mean_forecast = forecast.mean.values[-1]
    # Convertimos los retornos pronosticados a precios
    last_price = data[-1]
    predicted_prices = last_price * (1 + mean_forecast / 100).cumprod()
    return predicted_prices
