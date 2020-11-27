import numpy as np
import pandas as pd
from Class.QPSO import Q_PSO
# -*- coding: utf-8 -*-

container = np.load("pesos.npz")

data = [container[key] for key in container]
w1 = data[0]
w2 = data[1]
MSE = data[2]