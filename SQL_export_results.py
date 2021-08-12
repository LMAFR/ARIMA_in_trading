# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:29:26 2021

Este script se encarga de subir a MySQL los resultados obtenidos por las funciones de optimización y predicción.

@author: Usuario
"""

from __future__ import print_function
import pandas as pd
from sqlalchemy import create_engine 
import pymysql
import numpy as np
from datetime import datetime

## VOLCADO DE DATOS EN MYSQL

def carga_ARIMA(etiqueta, poblacion, prob_cruce, prob_muta_ind, prob_muta_gen, torneo, initial_date_train, end_date_train, p,d,q,\
                           initial_date_test, end_date_test, predictions, mse, smape, time_needed, test_data):
    
    '''
    
    Coloca los parámetros empleados en la optimización, la etiqueta (para conocer el par de divisas para el que se usaron) y los parámetros 
    sugeridos por el algoritmo de optimización para aplicarlos al modelo ARIMA.
    
    '''
    
    cadena_conexion= ('mysql+pymysql://root:#642l233X!@localhost:3306/ARIMA')
    conexion = create_engine(cadena_conexion)
    
    ## TABLA CON LOS PARÁMETROS Y RESULTADOS
    
    # etiqueta = "BTC-USD"
    
    dicc = {'ID': [etiqueta], 
            # 'ind_usado': np.array(ind_usado), # Realmente este dato no es necesario y me está dando problemas, así que lo descarto de momento.
            'poblacion': [poblacion],
            'prob_cruce': [prob_cruce],
            'prob_muta_ind': [prob_muta_ind],
            'prob_muta_gen': [prob_muta_gen],
            'torneo': [torneo],
            'initial_date_train': [initial_date_train],
            'end_date_train': [end_date_train],
            'p': [p],
            'd': [d],
            'q': [q],
            'initial_date_test': [initial_date_test],
            'end_date_test': [end_date_test],
            'predictions_key':['predictions_' + ('predictions_' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))],
            'mse':[round(mse, 4)],
            'rmse':[round(mse**0.5, 4)],
            'smape':[round(smape,4)],
            'time_needed': [round(time_needed/60, 3)]
            }
    
    df = pd.DataFrame(dicc)
    
    ## TABLA CON LAS PREDICCIONES
    
    df2 = pd.DataFrame(predictions, columns = ['Predictions']).round(decimals=2)
    df2['Datetime'] = test_data.index
    df2['Actual_data'] = test_data['Open'].round(decimals=2).values
    df2['ID'] = ('predictions_' + datetime.now().strftime("%d/%m/%Y_%H:%M:%S"))
    df2.head() #comprobación

    ## EXPORTACIÓN DE AMBAS TABLAS A SQL

    df.to_sql(name='parametros_y_resultados_ARIMA', con=cadena_conexion, if_exists= 'append') # Tabla de parámetros
    
    df2.to_sql(name='predicciones_ARIMA', con=cadena_conexion, if_exists= 'append') # Tabla de predicciones
    
    return print('The new parameters have been exported to the table "opt_parametros_y_resultados" (MySQL)')