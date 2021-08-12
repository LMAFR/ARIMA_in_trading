# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:34:44 2021

Hay que ejecutar cada función por separado primero y luego llamar a la función, si no se hace así dará un error en ajusta_y_predice_1.

@author: Usuario
"""

# Librerías

# Numpy
import numpy as np 
# Matplotlib
import matplotlib.pyplot as plt
# Pandas
import pandas as pd 
# Statsmodels
from statsmodels.tsa.arima.model import ARIMA
# Yfinance
import yfinance as yf 
# Plotly
#import plotly.graph_objects as go

# Time
import time

# Concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Genetic algorithms

import random
from deap import base, creator, tools, algorithms

# To locate the plots:
from IPython import get_ipython

# To accelerate the process:

def yf_data_ETL_7d1m(active_name): 
    
    '''
    Le pasas el nombre del activo y extrae un dataset con los valores del mismo para cada minuto durante 
    los últimos siete días. Ejemplos: 'EURUSD=X', 'BTC-USD', etc. Luego divide ese dataset en uno de 
    entrenamiento y otro testeo.
    '''
    
    data = yf.download(tickers=active_name, period='7d', interval='1m', progress=False)
    df = pd.DataFrame(data=data)
    train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
    
    return df, train_data, test_data

def ajusta_y_predice1(history, p, d, q):
    model = ARIMA(history, order=(p,d,q)) # p (lags) = 5, d (grado de diferenciación: corresponde a las d diferencias que son necesarias para convertir la serie de datos original en una estacionaria) = 1, q (orden de medias móviles usado) = 0. Tengo que justificar los dos últimos parámetros.
    model_fit = model.fit() # Entrenamos el modelo. disp = False indica que no hay que devolver un mensaje con los parámetros del modelo para cada iteración.
    output = model_fit.forecast() # Predecimos los siguientes valores a partir del modelo entrenado.
    return output[0]

def transforma_binario_a_integer(individuo):
   numero_binario = '0b'+str(individuo).replace('[','').replace(']','').replace(',','').replace(' ','')
   return (int(numero_binario, 2)) # OJO CON ESTA LINEA QUE ES CLAVE!

#print('El valor real para este momento fue: ', test_ar[0])
#print('La predicción es: ', prediccion[0])

# df, train_data, test_data = yf_data_ETL_7d1m('BTC-USD')




def optimiza_ARIMA_params(history, test_ar, poblacion = 32, prob_cruce = 0.5, prob_muta_ind = 0.4, prob_muta_gen = 0.4, torneo = 3):   
    
    start_time = time.time()
    
    def evalOneMin(individuo):
        
        # global history
        # global test_ar
        
        pbin = individuo[:3]
        dbin = individuo[3:5]
        qbin = individuo[5:7]
        
        p = transforma_binario_a_integer(pbin)
        d = transforma_binario_a_integer(dbin)
        q = 2 #transforma_binario_a_integer(qbin)
        
        print(' p = ',p, ', d = ', d, ', q = ', q, '\n\t#-- Done --#')
      
        prediction = ajusta_y_predice1(history, p, d, q)
        
        dif_to_min = np.abs(test_ar[0] - prediction) # Lo doy en tanto por 1 para tener más claro la diferencia de partida que poner (1 es mucho en este caso, por ejemplo).
      # dif_to_min
      #best_dif = 1 # Partimos de que la predicción sea 2 veces tan grande como la observación, lo que es una diferencia bastante grande comparada con las predicciones que he visto al aplicar el método.
     # if dif_to_min < best_dif:
    #      best_dif = dif_to_min
      
        return dif_to_min,
    
    # global history
    # global test_ar
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Observese que esto es un problema de minimizacion
    creator.create("Individual", list, fitness=creator.FitnessMin) # Observese que esto es un problema de minimizacion
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=7)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evalOneMin) # Registrar aqui la funcion de evaluacion
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=prob_muta_gen)
    toolbox.register("select", tools.selTournament, tournsize=torneo)
    
    def main():
        
        global history
        global test_ar
        
        #import numpy
        
        pop = toolbox.population(n=poblacion) # Para 16 valores de p, 8 de d y 1 de q, hay 128 combinaciones posibles.
        hof = tools.HallOfFame(1)
        
        # with ThreadPoolExecutor(8) as exe:
            
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=prob_cruce, mutpb=prob_muta_ind, ngen=3, stats=stats, halloffame=hof, verbose=True)
        
        return pop, logbook, hof
    
    # if __name__ == "__main__":
        
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print("El valor de p ha sido: ", transforma_binario_a_integer((hof[0])[0:3]))
    print("El valor de d ha sido: ", transforma_binario_a_integer((hof[0])[3:5]))
    print("El valor de q ha sido: ", 2) #transforma_binario_a_integer((hof[0])[5:7])
    print("--- %s seconds ---" % (time.time() - start_time))

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()
        
    ind_usado = hof[0]
    
    return ind_usado, poblacion, prob_cruce, prob_muta_ind, prob_muta_gen, torneo,\
        history, transforma_binario_a_integer((hof[0])[0:3]),\
        transforma_binario_a_integer((hof[0])[3:5]), 2 #transforma_binario_a_integer((hof[0])[5:7])
            

# ## Esta línea habría que cambiarla por la carga de datos desde SQL:
# df = pd.read_csv(r'C:\Users\Usuario\.spyder-py3\Trabajo_Data_Science\try1.csv')
# ##

# df.set_index('Datetime')
# train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
  
# train_ar = train_data['Open'].values
# history = [x for x in train_ar]

# # Creamos un array con los valores de testeo que vamos a usar:

# test_ar = test_data['Open'].values 

# combina = optimiza_ARIMA_params(history, test_ar)
# print(combina)

        # Es mejor no coger, a la vez, valores de p y de q mayores que 1. https://people.duke.edu/~rnau/411arim.htm
        
        # Combinaciones obtenidas para un dataset cambiante:
            
            # 6,2,0 128 ind 10 gen
            # 5,0,0 128 ind 10 gen
            # 14,3,0 64 ind 10 gen
            # 10,2,0 128 ind 5 gen
            # 8,1,0 64 ind 5 gen
            # 7,3,0 64 ind 5 gen
            # 3,2,0 64 ind 5 gen
            # 13,3,0 64 ind 5 gen            
            # 11,3,0 64 ind 5 gen            
            # 10,0,0 64 ind 5 gen            
            
        # Combinaciones obtenidas para un dataset fijo: (64 ind, 5 gen, 0.5 cruce, 0.4 prob mutar un individuo, 0.4 prob mutar un gen) -- 12 min
        
        # 1,0,0 -- 2.100
        # 1,0,0 -- 2.100
            
        # Combinaciones obtenidas para un dataset fijo: (32 ind, 3 gen, 0.5 cruce, 0.4 prob mutar un individuo, 0.4 prob mutar un gen)
        
        # 1,0,0 -- 2.100
        # 1,0,0 -- 2.100 -- 211s || 256s -- SMAPE: 1.041 -- MSE: 2096.48
        # 12,1,0 -- 6.549 -- 237s || 256s -- SMAPE: 1.041 -- MSE: 2096.48        
        
        