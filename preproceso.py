#!/usr/bin/env python
# coding: utf-8


"""
este script nos permite limpiar los datos y prepararlos
para ser utilizados en la red neuronal. En concreto se elimina
la columna 42 y se cambian los valores string a un parámetro
numérico.
"""


# librerias a utilizar
import numpy as np
import pandas as pd
from sklearn import preprocessing 


if __name__ == "__main__":



    # constantes para los archivos de entrada y salida
    DATA_PATH = 'Data/KDDTrain+_20Percent.txt'
    OUT_PATH = 'Data/train.txt'


    # lectura del dataset
    data = data = pd.read_csv(DATA_PATH, sep=",", header=None)
    # parametros normalización
    a = 0.1
    b = 0.99
    # randomizar la data
    data = data.sample(frac=1)


    # limpieza del dataset
    encoder = preprocessing.LabelEncoder()
    data[1]= encoder.fit_transform(data[1]) 
    data[2]= encoder.fit_transform(data[2]) 
    data[3]= encoder.fit_transform(data[3]) 
    # se elimina la última columna (ya que esta contiene la complejidad)
    data = data.drop(42,axis = 1)
    data = data.drop(19,axis = 1)
    data = data.drop(20,axis = 1)

    # remplazar columan 41. esta contiene el resultado
    # normal = 1, cualquier otro caso = -1
    data[41] = data[41]=='normal'
    data[41] = data[41].replace(True, 1)
    data[41] = data[41].replace(0, -1)


    # se toma la matriz de entrada (todas menos la 41)
    X = data.loc[:, data.columns != 41]
    # se toma vector con los resultados (41)
    y = data.loc[:, data.columns == 41]
    # se convierte la matriz de entrada a un vector


    normalized_X=(X-X.min())/(X.max()-X.min())
    normalized_X = (b-a)*normalized_X + a 
    
    # se genera un nuevo dataframe con los valores normalizados
    normalized_X = pd.DataFrame(normalized_X)
    output_data = normalized_X
    output_data = output_data.join(y)







    output_data.to_csv(path_or_buf = 'Data/train.txt', index = False    , mode = 'w+')

    # se procede a escribir el nuevo dataframe como train.txt
    output_data.to_csv(path_or_buf = OUT_PATH, index = False, mode = 'w+')







