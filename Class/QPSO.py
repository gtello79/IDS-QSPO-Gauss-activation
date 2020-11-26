import random as rd
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse

class Q_PSO:
    
    def __init__(self):
        self.np = None
        self.nh = None
        self.weight = None
        self.X = None
        #Falta algo que le de el valor a las iteraciones, ojo al tejo
        self.maxIter = None
        self.D = None #numero de columnas del input 
        self.ye = None
        self.C = None
        
    def ini_swarm(self, num_part, num_hidden, D):  #D es el la dimension del input en este caso 40 app
        X = np.zeros((num_part,num_hidden*D),dtype=float) 
        self.np = num_part
        self.nh = num_hidden

        for i in range(num_part):
            #Weight hidden NP
            wh = self.rand_w(num_hidden,D) #retorna una matriz de tamaño num_hidden*D
            a = np.reshape(wh, (1, num_hidden*D))
            X[i]= a
        self.X = X #no se si era lo mismo que weight
    
    def rand_w(self, nextNodes, currentNodes):
        w = np.random.random((nextNodes, currentNodes))
        x = nextNodes+currentNodes
        r = np.sqrt(6/x)
        w = w*2*r - r
        return w
    
    def gaussian_activation(self, x_n, w_j):
        z = np.linalg.norm(x_n - w_j) #en el codigo el profe recibe solo x_n no se de donde saca w_j
        return math.exp(-0.5*z*z)
        
        
    def run_QPSO(self):
        return True

    def fitness(self, num_hidden, D, num_part, X, xe, ye): 
        w2 = np.zeros((num_hidden, D), dtype=float)
        MSE = np.zeros((num_part, 1), dtype=float)
        for i in range(num_part):
            p = X[i]
            w1 = np.reshape(X, (num_hidden, D)) #se vuelve a estructurar como matriz
            H = self.gaussian_activation(xe, w1)
            w2[i] = self.mlp_pinv(H)
            ze = w2[i]*H
            MSE[i] = math.sqrt(mse(ye, ze))
        return(MSE, w2)
    
    def fitness_no_arg(self):  #lo mismo sin argumentos
        w2 = np.zeros((self.nh, self.D), dtype=float)
        MSE = np.zeros((self.np, 1), dtype=float)
        for i in range(self.np):
            p = self.X[i]
            w1 = np.reshape(self.X, (self.nh, self.D)) #se vuelve a estructurar como matriz
            H = self.gaussian_activation(self.xe, w1)
            w2[i] = self.mlp_pinv(H)
            ze = w2[i]*H
            MSE[i] = math.sqrt(mse(self.ye, ze))
        return(MSE, w2)
    
    def mlp_pinv(self, H):
        L,N = H.shape
        yh = self.ye*np.transpose(H)
        hh = (H*np.transpose(H) + np.eye(L)/self.C)
        w2 = yh*np.linalg.pinv(hh)
        return w2
        
        
    
test = Q_PSO()
