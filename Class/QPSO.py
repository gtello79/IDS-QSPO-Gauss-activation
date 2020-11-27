#!/usr/bin/env python
import random as rd
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse

class Q_PSO:

    def __init__(self, maxIter, numPart, numHidden, D, xe, ye):
        self.maxIter = maxIter
        self.np = numPart
        self.nh = numHidden
        self.weight = None
        self.X = None
        self.D = D
        #Inicializar xe - ye
        self.xe = xe  
        self.ye = ye
        self.C = None

        #Inicializacion de la poblacion
        self.ini_swarm(numPart,numHidden,D)
        self.fitness_no_arg()


    def ini_swarm(self, num_part, num_hidden, D):
        self.np = num_part
        self.nh = num_hidden
        
        dim = self.nh*D  
        X = np.zeros( (self.np, dim), dtype=float) 
        
        for i in range(self.np):
            wh = self.rand_w(self.nh, D)
            a = np.reshape(wh, (1, dim))
            X[i]= a
        self.X = X 
        

    def rand_w(self, nextNodes, currentNodes):
        w = np.random.random((nextNodes, currentNodes))
        x = nextNodes+currentNodes
        r = np.sqrt(6/x)
        w = w*2*r - r
        return w
    
    #funcion de activacion
    def gaussian_activation(self, x_n, w_j):
        z = np.matmul(w_j,np.transpose(x_n))
        for number in z:
            number = np.exp(-1*(number*number))
        return z
        
        
    def run_QPSO(self):
        for i in range(self.maxIter):
            newPFitness, newBeta = self.fitness_no_arg()
            print
            #newPFitness, newBeta = self.fitness(self.xe, self.ye, self.nh, self.X, self.)
            
        return True

    #esta funcion lo que hace mas o menos es recomponer las matrices de pesos de cada particula
    #y testear su MSE

    def fitness(self):  #lo mismo sin argumentos
        w2 = np.zeros((self.np, self.nh), dtype=float)
        MSE = np.zeros(self.np , dtype=float)
        for i in range(self.np):
            p = self.X[i]
            w1 = np.reshape(p, (self.nh, self.D)) #se vuelve a estructurar como matriz
            H = self.gaussian_activation(self.xe, w1)
            w2[i] = self.mlp_pinv(H)
            ze = np.matmul(w2[i],H)
            MSE[i] = math.sqrt(mse(self.ye, ze))
        return MSE, w2
    
    def mlp_pinv(self, H):
        L,N = H.shape

        yh = np.matmul(np.transpose(self.ye),np.transpose(H))

        hh = np.matmul(H,np.transpose(H)) #falta agregar eye/C

        w2 = np.matmul(np.transpose(yh),np.linalg.pinv(hh))
       # print(w2.shape)
        return w2
    
    #7 argumentos xdxd
    def upd_particle(self, X, pBest,pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):
        for i in range(pFitness):
            if (New_pFitness[i] < pFitness[i]):
                pFitness[i] = New_pFitness[i]
                pBest[i][:] = X[i, :]
        New_gFitness = min(pFitness)
        idx = np.argmin(pFitness)
        if (New_gFitness < gFitness):
            gFitness = New_gFitness
            gBest = pBest[idx][:]
            wBest = newBeta[idx][:]
            
        return pBest, pFitness, gBest, gFitness, wBest
        
        
