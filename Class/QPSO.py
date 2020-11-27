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
        self.run_QPSO()


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
        z = np.linalg.norm(x_n - w_j)
        return math.exp(-0.5*z*z)
        
        
    def run_QPSO(self):
        for i in range(self.maxIter):
            newPFitness, newBeta = self.fitness_no_arg()
            print
            #newPFitness, newBeta = self.fitness(self.xe, self.ye, self.nh, self.X, self.)
            
        return True

    #esta funcion lo que hace mas o menos es recomponer las matrices de pesos de cada particula
    #y testear su MSE
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
        return MSE, w2
    
    def fitness_no_arg(self):  #lo mismo sin argumentos
        w2 = np.zeros((self.nh, self.D), dtype=float)
        MSE = np.zeros(self.np , dtype=float)
        for i in range(self.np):
            p = self.X[i]
            w1 = np.reshape(p, (self.nh, self.D)) #se vuelve a estructurar como matriz
            H = self.gaussian_activation(self.xe, w1)
            w2[i] = self.mlp_pinv(H)
            ze = w2[i]*H
            MSE[i] = math.sqrt(mse(self.ye, ze))
        return MSE, w2
    
    def mlp_pinv(self, H):
        L,N = H.shape
        yh = self.ye*np.transpose(H)
        hh = (H*np.transpose(H) + np.eye(L)/self.C)
        w2 = yh*np.linalg.pinv(hh)
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
        
        
