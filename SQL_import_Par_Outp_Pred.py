# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 09:19:37 2021

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:27:49 2021

Este script pretende se encarga de obtener los datos guardados en la base de datos MySQL para luego usarlos en las funciones de optimización, ajustes
de modelos y predicciones.

@author: Usuario
"""
from __future__ import print_function
import pandas as pd
import pymysql

## LECTURA DE PARAMETROS Y RESULTADOS DESDE PYTHON

def importa_parametros_y_resultados(etiqueta):

    query = '''
    select *
    from ARIMA.parametros_y_resultados_arima
    '''
    conn = pymysql.connect(
                host = 'localhost',
                user= '****',
                password= '****',
                db = 'ARIMA'
            )
    
    data = pd.read_sql(query, conn)
    
    df = data[data['ID'] == etiqueta]
    return df

## LECTURA DE PREDICCIONES PARA UNOS PARÁMETROS Y RESULTADOS CONCRETOS

def importa_predicciones(pred_ID):

    query = '''
    select *
    from ARIMA.predicciones_arima
    '''
    conn = pymysql.connect(
                host = 'localhost',
                user= '****',
                password= '****',
                db = 'ARIMA'
            )
    
    data = pd.read_sql(query, conn)
    
    df = data[data['ID'] == pred_ID]
    return df

# df = importa_datos('BTC-USD')
# print(df)