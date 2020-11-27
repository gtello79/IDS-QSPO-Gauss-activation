#!/usr/bin/env python
import random as rd
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error as mse

class Q_PSO:

    def __init__(self, maxIter, numPart, numHidden, D, xe, ye, C):
        self.maxIter = maxIter
        self.np = numPart
        self.nh = numHidden
        self.weight = None
        self.X = None
        self.D = D
        #Inicializar xe - ye
        self.xe = xe  
        self.ye = ye
        self.C = C

        #Inicializacion de la poblacion
        self.ini_swarm(numPart,numHidden,D)
        self.gBest, self.wBest, self.mBest = self.run_QPSO()


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
        iter = 0
        alfa = np.zeros(maxIter)
        for p in range(maxIter):
            alfa[p] = 0.95 - ((0.95 - 0.2)/self.maxIter)*p
        pBest = np.zeros((self.np, self.D*self.nh))
        pFitness = np.ones((1,self.np))*100000
        gBest = np.ones((1, self.D*self.nh))
        wBest = np.zeros((1,self.nh))
        gFitness = 1000000000
        MSE = np.zeros((self.maxIter))
        
        for iter in range(self.maxIter):
            new_pFitness, newBeta = self.fitness()
            pBest, pFitness, gBest, gFitness, wBest = self.upd_particle(self.X, pBest, pFitness, gBest,
                                                                gFitness,new_pFitness, newBeta, wBest)

            MSE[iter] = gFitness
            mBest = pBest.mean(axis=0)
            for i in range(self.np):
                for j in range(self.nh*self.D):
                    phi=rd.random()
                    u = rd.random()
                    pBest[i][j] = phi*pBest[i][j] + (1-phi)*gBest[j]
                    if (rd.random() > 0.5):
                        self.X[i][j] = pBest[i][j] + alfa[iter]*abs(mBest[j]-self.X[i][j])*math.log(1/u)
                    else:
                        self.X[i][j] = pBest[i][j] - alfa[iter]*abs(mBest[j]-self.X[i][j])*math.log(1/u)
                        
        return gBest, wBest, MSE
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

        hh = np.matmul(H,np.transpose(H))
        hh = hh + (np.eye(4)/self.C)


        w2 = np.matmul(np.transpose(yh),np.linalg.pinv(hh))

        return w2
    

    def upd_particle(self, X, pBest,pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):
        
        for i in range(self.np):
            if (New_pFitness[i] < pFitness[0][i]):
                pFitness[0][i] = New_pFitness[i]
                pBest[i][:] = X[i, :]
        New_gFitness = min(pFitness[0])
        idx = np.argmin(pFitness)
        if (New_gFitness < gFitness):
            gFitness = New_gFitness
            gBest = pBest[idx][:]
            wBest = newBeta[idx][:]

        return pBest, pFitness, gBest, gFitness, wBest
        
maxIter = 300
numPart = 5
numHidden = 4
DATA_PATH = "../DATA/test.txt"
data = pd.read_csv(DATA_PATH)

xe = data.iloc[:, 1:40]
ye = data.iloc[:, 40]
    
D, N = xe.shape

L = 20
C = 2

xe = np.array(xe)
ye = np.array(ye)

X0 = np.ones((D,1))
Xe = np.hstack((xe, X0))
    
N = N+1


q = Q_PSO(maxIter, numPart, numHidden, N, Xe, ye, C)      
