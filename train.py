import numpy as np
import pandas as pd



DATA_PATH = "DATA/test.txt"
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

