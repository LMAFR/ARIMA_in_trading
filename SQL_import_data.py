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

## LECTURA DE DATOS DESDE PYTHON

def importa_datos(etiqueta):

    query = '''
    select ID, Datetime, Open, High, Low, Close, Adj_close, Volume
    from TRADING2.activos
    '''
    conn = pymysql.connect(
                host = 'localhost',
                user= '****',
                password= '****',
                db = 'TRADING2'
            )
    
    data = pd.read_sql(query, conn)
    
    df = data[data['ID'] == etiqueta].iloc[:,1:].set_index('Datetime')
    return df

# df = importa_datos('BTC-USD')
# print(df)
