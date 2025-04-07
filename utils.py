import numpy as np
import pandas as pd

def load_data_from_csv(filepath):
    """
    Carga datos de un archivo CSV.
    :param filepath: Ruta del archivo CSV.
    :return: DataFrame con los datos cargados.
    """
    return pd.read_csv(filepath)

def normalize_data(data):
    """
    Normaliza los datos a un rango de 0 a 1.
    :param data: Array de datos a normalizar.
    :return: Array de datos normalizados, valor mínimo y valor máximo.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    norm_data = (data - data_min) / (data_max - data_min)
    return norm_data, data_min, data_max

def denormalize_data(norm_data, data_min, data_max):
    """
    Desnormaliza los datos al rango original.
    :param norm_data: Array de datos normalizados.
    :param data_min: Valor mínimo original.
    :param data_max: Valor máximo original.
    :return: Array de datos desnormalizados.
    """
    return norm_data * (data_max - data_min) + data_min
