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
        self.X = None
        self.D = D
        #Inicializar xe - ye
        self.xe = xe  
        self.ye = ye
        self.C = C
        self.w1 = None

        #Inicializacion de la poblacion
        self.ini_swarm(numPart,numHidden,D)



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
        alfa = np.zeros(self.maxIter)
        for p in range(self.maxIter):
            alfa[p] = 0.95 - ((0.95 - 0.2)/self.maxIter)*p
        pBest = np.zeros((self.np, self.D*self.nh))
        pFitness = np.ones(self.np)*100000
        gBest = np.ones(self.D*self.nh)
        wBest = np.zeros(self.nh)
        gFitness = 1000000000
        MSE = np.zeros((self.maxIter))

        for iter in range(self.maxIter):
            print("Iteracion numero "+ str(iter+1))
            new_pFitness, newBeta = self.fitness()
            pBest, pFitness, gBest, gFitness, wBest = self.upd_particle(self.X, pBest, pFitness, gBest,
                                                                gFitness,new_pFitness, newBeta, wBest)

            MSE[iter] = gFitness

            mBest = pBest.mean(axis=0)
            avg_t = 0
            for i in range(self.np):
                for j in range(self.nh*self.D):
                    phi=rd.random()
                    u = rd.random()
                    pBest[i][j] = phi*pBest[i][j] + (1-phi)*gBest[j]
                    t = alfa[iter]*abs(mBest[j]-self.X[i][j])*math.log(1/u)
                    avg_t += t
                    if (rd.random() > 0.5):
                        self.X[i][j] = pBest[i][j] + t

                    else:
                        self.X[i][j] = pBest[i][j] - t
            avg_t = avg_t/(self.np*self.nh*self.D)

        self.w1 = np.reshape(gBest, (self.nh, self.D))
        return gBest, wBest, MSE


    #esta funcion lo que hace mas o menos es recomponer las matrices de pesos de cada particula
    #y testear su MSE

    def fitness(self):  
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
        hh = hh + (np.eye(hh.shape[0])/self.C)
        w2 = np.matmul(np.transpose(yh),np.linalg.pinv(hh))

        return w2
    

    def upd_particle(self, X, pBest,pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):
        
        for i in range(self.np):
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

