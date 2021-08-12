# Numpy
import numpy as np
# Matplotlib
import matplotlib.pyplot as plt
# Pandas
import pandas as pd
from pandas.plotting import lag_plot
from pandas import datetime
# Statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
# Time
import time
# Yfinance
import yfinance as yf
# Plotly
import plotly.graph_objects as go

from IPython import get_ipython

# Concurrent.futures
from concurrent.futures import ThreadPoolExecutor

## Funciones

def yf_data_ETL_7d1m(active_name):

    '''
    Le pasas el nombre del activo y extrae un dataset con los valores del mismo para cada minuto durante
    los últimos siete días. Ejemplos: 'EURUSD=X', 'BTC-USD', etc. Luego divide ese dataset en uno de
    entrenamiento y otro testeo.
    '''

    data = yf.download(tickers=active_name, period='7d', interval='1m')
    df = pd.DataFrame(data=data)
    train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]

    return df, train_data, test_data

def ARIMA_checklist(df, train_data, test_data, lags):

    '''
    Esta función lleva a cabo una serie de comprobaciones que nos dirán si es lógico usar ARIMA,
    cuáles son los valores que tradicionalmente se predecirían para p,d y q, ... También muestra
    el gráfico con los datos para entrenar y los datos de testeo.

    df: dataframe con los datos a emplear en el modelo.
    train_data: subconjunto de datos de dicho dataframe que se usarán en el entrenamiento.
    test_data: subconjunto de datos de dicho dataframe que se usarán para comprobar cómo de bien funciona el modelo.
    lags: un número entero (>0)

    '''

    ## Gráfico con los datos empleados en el entrenamiento y testeo:

    plt.figure(figsize=(12,7))
    plt.title('Open Prices')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.plot(train_data['Open'], 'blue', label='Training Data')
    plt.plot(test_data['Open'], 'green', label='Testing Data')
    plt.legend()
    plt.show()

    ## Gráfico de aurocorrelación:

    plt.figure(figsize=(10,10))
    lag_plot(df['Open'], lag=lags) # Número de minutos necesarios para que el precio vuelva a ser el mismo que era previamente
                                   # (se aplica para cada fecha guardada en nuestro dataset).
    plt.title('Autocorrelation plot')
    plt.show()

    ## Gráfico para diferenciación de grado 1:

    df_stocks_diff = train_data['Open']-train_data['Open'].shift() # Resto a cada valor de stock_diff el valor del índice
#                                                                    anterior (si no había, devuelve NaN)
    df_stocks_diff.plot() # La idea es que con esto hacemos que la serie temporal se pueda considerar estacionaria (se pueden
#                           hacer restas de mayor orden (más complejas), esta es la correspondiente a d = 1.
    plt.show()

    ## Otros gráficos de autocorrelación usados para dar un valor aproximado de p y q:

    lag_acf = acf(train_data['Open'], nlags = lags) # Plot de la autocorrelación para 10 lags.
    lag_pacf = pacf(train_data['Open'], nlags = lags, method = 'ols') # Plot de la autocorrelación parcial para 10 lags.

    plt.figure(figsize=(13,6))
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y = 0, linestyle = '--', color = 'gray')
    plt.axhline(y = -1.96/np.sqrt(len(df_stocks_diff)), linestyle = '--', color = 'gray')
    plt.axhline(y = +1.96/np.sqrt(len(df_stocks_diff)), linestyle = '--', color = 'gray')
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y = 0, linestyle = '--', color = 'gray')
    plt.axhline(y = -1.96/np.sqrt(len(df_stocks_diff)), linestyle = '--', color = 'gray')
    plt.axhline(y = +1.96/np.sqrt(len(df_stocks_diff)), linestyle = '--', color = 'gray')
    plt.show()

def ajusta_y_predice_todo(df, train_data, test_data, p, d, q):

    '''
    Esta función va a ajustar el modelo ARIMA a los datos del entrenamiento y los usará para predecir el siguiente valor
    (que será el primero de nuestro dataframe de datos de testeo). Una vez predicho, el valor del dataframe de testeo se
    transfiere al de entrenamiento, se vuelve a ajustar el modelo y se predice el siguiente. De esta forma, el modelo va
    prediciendo un resultado para cada observación en el conjunto de datos de testeo y, al final obtenemos una representación
    gráfica de los datos de testeo frente a los datos predichos.

    El hecho de ir añadiendo los datos de testeo al dataframe de entrenamiento es realista, ya que en este tipo de sistemas lo
    ideal es incorporar en tiempo real cada nueva observación al dataframe antes de predecir el siguiente resultado (al menos
    en el caso de que los tiempos de ejecución lo permitan). En nuestro caso, al ser intervalos de 1 minuto y tratarse de un
    modelo relativamente simple, los tiempos de ejecución no son un problema.

    '''

    def ajusta_y_predice1(history, p, d, q):
        model = ARIMA(history, order=(p,d,q)) # p (lags) = 5, d (grado de diferenciación: corresponde a las d diferencias que son necesarias para convertir la serie de datos original en una estacionaria) = 1, q (orden de medias móviles usado) = 0. Tengo que justificar los dos últimos parámetros.
        model_fit = model.fit() # Entrenamos el modelo. disp = False indica que no hay que devolver un mensaje con los parámetros del modelo para cada iteración.
        output = model_fit.forecast() # Predecimos los siguientes valores a partir del modelo entrenado.
        return output[0]

    def smape_kun(y_true, y_pred):
        '''
        Se define una función para calcular SMAPE, un tipo de error bastante usado para cuantificar cómo de bueno es el
        ajuste en modelos predictivos como ARIMA.
        '''
        return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

    start_time = time.time()

    # Creamos una lista con los valores del precio de apertura para aplicarle el modelo ARIMA
    # (no lo hacemos directamente con el dataset para poder meterle luego más datos):

    train_ar = train_data['Open'].values
    history = [x for x in train_ar]

    # Creamos un array con los valores de testeo que vamos a usar:

    test_ar = test_data['Open'].values

    # Creamos una lista donde iremos almacenando las predicciones:

    predictions = list()

    # Aplicamos ARIMA prediciendo un valor para cada iteración del bucle for, tal y como se ha explicado en el markdown:

    with ThreadPoolExecutor(8) as exe:

        for t in range(len(test_ar)):
            yhat = ajusta_y_predice1(history, p, d, q)
            predictions.append(yhat) # Añadimos dicho valor a la lista "predictions"
            obs = test_ar[t] # Almacenamos el valor de la observación que corresponde al valor predicho (que es del dataset de testeo) en una variable.
            history.append(obs) # Añadimos dicho valor a la lista con los datos de entrenamiento, para que cuando se prediga el próximo valor, el dataset sea un poco más grande.

    error = mean_squared_error(test_ar, predictions) # Calculamos el error cuadrático medio entre las observaciones de testeo y los valores predichos.
    print('Testing Mean Squared Error: %.3f' % error)
    
    error2 = smape_kun(test_ar, predictions) # También calculamos el SMAPE
    print('Symmetric mean absolute percentage error: %.3f' % error2)
    
    time_needed = (time.time() - start_time)
    print("--- %s seconds ---" % time_needed)

    return predictions, error, error2, time_needed

def compara_prediccion(df, test_data, predictions):

    '''
    Dada una lista de predicciones obtenidas de aplicar la función ARIMA_train_and_pred(), esta función te representa las
    predicciones junto a las observaciones reales en un plotly, de tal forma que puedas hacer zoom en las gráficas para
    comprobar como de buena es realmente la predicción a nivel visual. Para ello, como es lógico, también necesita el
    dataframe con las observaciones de partida y el dataframe con el suconjunto de observaciones que se han escogido para
    comprobar lo buenas que son las predicciones (datos para testeo).
    '''

    predictions_df = pd.DataFrame(predictions, columns=['Predictions']).set_index(test_data.index)

    fig = go.Figure()

    # Add traces

    plt.figure(figsize=(12,7))
    
    get_ipython().run_line_magic('matplotlib', 'qt')

    fig.add_trace(go.Scatter(x=df.index, y=df['Open'],
                        mode='lines',
                        name='Training Data'))
    fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Predictions'],
                        mode='lines+markers',
                        name='Predicted Price'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Open'],
                        mode='lines+markers',
                        name='Actual Price'))

    fig.show()

# combinaciones = [[5,1], [6,2], [5,0], [14,3], [10,2], [8,1], [7,3], [3,2], [13,3], [11,3], [10,0]]

# df, train_data, test_data = yf_data_ETL_7d1m('BTC-USD')

# # for elemento in combinaciones:
# ajusta_y_predice_todo(df, train_data, test_data, 2, 1, 2)


# 1.076-97 3240-90
# 