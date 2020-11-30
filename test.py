import numpy as np
import pandas as pd
import time
# -*- coding: utf-8 -*-

start_time = time.time()


def gaussian_activation(x_n, w_j):
    z = np.matmul(w_j,np.transpose(x_n))
    for number in z:
        number = np.exp(-1*(number*number))
    return z
        
def metrica(y_esperado, y_obtenido):
    output = open("Data/metricas.txt", "a")
    vp = 0
    fp = 0
    fn = 0
    vn = 0
    
    for i in range(len(y_esperado)):
        if y_esperado[i] == 1 and y_obtenido[i] == 1:
            vp += 1
        if y_esperado[i] == -1 and y_obtenido[i] == -1:
            vn += 1
        if y_esperado[i] == 1 and y_obtenido[i] == -1:
            fn += 1
        if y_esperado[i] == -1 and y_obtenido[i] == 1:
            fp += 1
    accuracy_normal = vp/(vp+fp)
    recall_normal = vp/(vp+fn)
    f_score_normal = 2*(accuracy_normal*recall_normal/(accuracy_normal+recall_normal))
    accuracy_ataque = vn/(vn+fn)
    recall_ataque = vn/(vn + fp)
    f_score_ataque = 2*(accuracy_ataque*recall_ataque/(accuracy_ataque + recall_ataque))
    
    print("Accuracy = "+ str(accuracy_normal))
    print("F_score normal = " + str(f_score_normal))
    print("F_score ataque = " + str(f_score_ataque))    
    output.write("Accuracy: "+ str(accuracy_normal) + ", F score normal: " + str(f_score_normal) + 
                 ", F score ataque: " + str(f_score_ataque) + "\n")
    output.close()
 

      
    return accuracy_normal, f_score_normal, f_score_ataque

#cargar data para test
DATA_PATH = "Data/test.txt"
data = pd.read_csv(DATA_PATH)

xe = data.iloc[:, 1:-1] 
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


#se transforma el valor dependiendo de si es mayor o menor a 0 en prediccion
for number in range(len(zv)):
    if zv[number] < 0: 
        zv[number] = -1
    else:
        zv[number] = 1

#Utilizar nuestras propias metricas



print("Tiempo de test: %s segundos" % (time.time() - start_time))

accuracy, f_score_normal, f_score_ataque = metrica(ye, zv)