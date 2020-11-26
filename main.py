#!/usr/bin/python3
import pandas as pd
from Class.QPSO import Q_PSO
 
if __name__ == "__main__":
    DATA_PATH = "DATA/test.txt"
    data = pd.read_csv(DATA_PATH)
    q = Q_PSO()
