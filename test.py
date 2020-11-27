import numpy as np
import pandas as pd
from Class.QPSO import Q_PSO
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# -*- coding: utf-8 -*-

def gaussian_activation(x_n, w_j):
    z = np.matmul(w_j,np.transpose(x_n))
    for number in z:
        number = np.exp(-1*(number*number))
    return z

#cargar data para test
DATA_PATH = "Data/kddtest.txt"
data = pd.read_csv(DATA_PATH)

xe = data.iloc[:, 1:-1] #probando sample mas chico
ye = data.iloc[:, -1]

N, D = xe.shape
xe = np.array(xe)
ye = np.array(ye)

X0 = np.ones((N,1))
Xe = np.hstack((xe, X0))



#cargar pesos entrenados
container = np.load("pesos.npz")

PARAM_CONFIG_PATH = "param_config.csv"
params = pd.read_csv(PARAM_CONFIG_PATH)
L = params.iloc[0,0]
C = params.iloc[0,1]
maxIter = params.iloc[0,2]
numPart = params.iloc[0,3]

weight_data = [container[key] for key in container]
w1 = weight_data[0]
w2 = weight_data[1]
MSE = weight_data[2]

w1 = w1.reshape((L,D+1))
H = gaussian_activation(Xe, w1)
zv = np.matmul(w2,H)

#supongo que hay que transformar dependiendo del valor que arroje
for number in range(len(zv)):
    if zv[number] < 0: 
        zv[number] = -1
    else:
        zv[number] = 1

#No creo que se pueda usar sklearn para esto y creo que no es lo mismo q pide el profe xd
f_score = f1_score(ye, zv, average='macro')
accuracy = accuracy_score(ye, zv)
