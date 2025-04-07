import numpy as np

def ensemble_prediction(preds, weights=None):
    """
    Combina las predicciones de varios modelos.
    :param preds: Lista de arrays (predicciones) de igual longitud.
    :param weights: Lista de pesos para cada modelo. Si no se especifica, se utiliza el promedio simple.
    :return: Array con la predicción combinada.
    """
    preds = np.array(preds)
    if weights is None:
        return np.mean(preds, axis=0)
    else:
        weights = np.array(weights) / np.sum(weights)
        return np.average(preds, axis=0, weights=weights)
