import random as rd
import numpy as np

class Q_PSO:
    
    def __init__(self):
        self.np = None
        self.nh = None
        self.weight = None
        #Falta algo que le de el valor a las iteraciones, ojo al tejo
        self.maxIter = None
        
    def ini_swarm(self, nodesP, nodesH, D):
        matrix = np.zeros((NP,NH),dtype=float)
        self.np = nodesP
        self.nh = nodesH

        for i in range(NP):
            #Weight hidden NP
            wh = self.rand_w(NH,D)
            matrix[i][:]= wh
        self.weight = matrix
    
    def rand_w(self, nextNodes, currentNodes):
        w = rd.uniform(nextNodes, currentNodes)
        x = nextNodes+currentNodes
        r = np.sqrt(6/x)
        w = w*2*r - r
        return w

    def run_QPSO(self):
        return True

    def fitness(self):
        return True