# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:55:19 2021

Este archivo tiene como finalidad automatizar el proceso de importar los datos de SQL, usarlos en la optimización de los parámetros del modelo
ARIMA y luego usarlos junto con los parámetros optimizados en el modelo ARIMA.

Este script está diseñado para tratar con el dataset estático, que se dividirá en uno de entrenamiento y otro de testeo para devolver el resultado final.

@author: AFR
"""

## LIBRERIAS EMPLEADAS

# Pandas
from datetime import datetime
import time

# import pandas as pd # Una vez se reemplace el dataset fijo por la importación de datos desde SQL, esto ya no debería hacer falta.

## CARGA DE LOS SCRIPTS QUE CONTIENEN LAS FUNCIONES A USAR

from ARIMA_optim_alggen import optimiza_ARIMA_params
from funciones_ARIMA import ajusta_y_predice_todo
from funciones_ARIMA import compara_prediccion
from SQL_import_data import importa_datos
from SQL_export_results import carga_ARIMA

## FUNCION DE CARGA DE DATOS (Python + query de SQL)

# Esto habría que reemplazarlo por la query que se trae el mismo tipo de dataframe
etiqueta = 'BTC-USD'
df = importa_datos(etiqueta)
# df = pd.read_csv(r'C:\Users\Usuario\.spyder-py3\Trabajo_Data_Science\try1.csv')
# ------------------------------------------------------------------------------

# Estas líneas son para transformar el dataframe a la forma que recibirá la función de algoritmos genéticos:

# df.set_index('Datetime')
train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]

train_ar = train_data['Open'].values
history = [x for x in train_ar]

initial_date_train = train_data.index[0]
end_date_train = train_data.index[-1]

initial_date_test = test_data.index[0]
end_date_test = test_data.index[-1]

train_data['Open']
test_ar = test_data['Open'].values

while True:

    ## FUNCION DE OPTIMIZACION DE PARÁMETROS

    ind_usado, poblacion, prob_cruce, prob_muta_ind, prob_muta_gen, torneo,\
        history,p,d,q = optimiza_ARIMA_params(history, test_ar)
    print(p,d,q)

    ## FUNCION DE AJUSTE Y PREDICCIÓN DEL MODELO

    predictions, e_mse, e_smape, time_needed = ajusta_y_predice_todo(df, train_data, test_data, p, d, q)

    ## FUNCIÓN DE EXPORTACIÓN DE LOS PARÁMETROS Y RESULTADOS DE LA OPTIMIZACIÓN Y EL MODELO

    carga_ARIMA(etiqueta, poblacion, prob_cruce, prob_muta_ind, prob_muta_gen, torneo, initial_date_train, end_date_train, p, d, q,\
                           initial_date_test, end_date_test, predictions, e_mse, e_smape, time_needed, test_data)

    time.sleep(300)

## ELABORACIÓN DE LA GRÁFICA DE PREDICCIONES VS. DATOS REALES

# compara_prediccion(df, test_data, predictions) # No me saca la gráfica tipo go. Probar con un plot normal primero.
