# lstm_model.py
#import numpy as np
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.optimizers import Adam
#
#def create_dataset(data, look_back=3):
#    X, y = [], []
#    for i in range(len(data) - look_back):
#        X.append(data[i:i+look_back])
#        y.append(data[i+look_back])
#    return np.array(X), np.array(y)
#
#def predict(data, steps=10, look_back=3, epochs=50):
#    """
#    Entrena un modelo LSTM simple y predice valores futuros.
#    :param data: Serie de tiempo (lista o array) de precios.
#    :param steps: Número de períodos a predecir.
#    :param look_back: Tamaño de la ventana para el LSTM.
#    :param epochs: Número de épocas de entrenamiento.
#    :return: Array con la predicción.
#    """
#    data = np.array(data)
#    # Normalizar los datos
#    data_min = np.min(data)
#    data_max = np.max(data)
#    norm_data = (data - data_min) / (data_max - data_min)
#    
#    X, y_data = create_dataset(norm_data, look_back)
#    X = X.reshape((X.shape[0], X.shape[1], 1))
#    
#    # Crear el modelo LSTM
#    model = Sequential([
#        LSTM(50, activation='relu', input_shape=(look_back, 1)),
#        Dense(1)
#    ])
#    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
#    model.fit(X, y_data, epochs=epochs, verbose=0)
#    
#    # Predecir de forma iterativa
#    last_sequence = norm_data[-look_back:].tolist()
#    predictions = []
#    for _ in range(steps):
#        seq = np.array(last_sequence).reshape((1, look_back, 1))
#        pred_norm = model.predict(seq, verbose=0)[0,0]
#        predictions.append(pred_norm)
#        last_sequence.pop(0)
#        last_sequence.append(pred_norm)
#    
#    forecast = np.array(predictions) * (data_max - data_min) + data_min
#    return forecast
#