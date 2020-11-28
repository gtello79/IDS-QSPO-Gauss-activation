#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
from sklearn import preprocessing #no se si se podra usar esto para label encoder xd


if __name__ == "__main__":

    def norma(f_vector):
        x_max = max(f_vector)
        x_min = min(f_vector)

        for i in range(len(f_vector)):
            y = (f_vector[i] - x_min) / (x_max - x_min)
            y = (b - a) * y + a
            f_vector[i] = y

        return f_vector
    
    
    
    def scale_features(data):
        for i in range(len(data)):
            data[i] = norma(data[i])

        return data

    DATA_PATH = 'Data/KDDTrain+_20Percent.txt'
    data = data = pd.read_csv(DATA_PATH, sep=",", header=None)
    a = 0.1
    b = 0.99
    

    data = data.sample(frac=1) #randomizar la data

    encoder = preprocessing.LabelEncoder()

    #pero que eficiente xdxd
    data[1]= encoder.fit_transform(data[1]) 
    data[2]= encoder.fit_transform(data[2]) 
    data[3]= encoder.fit_transform(data[3]) 
    



    
    data = data.drop(42,axis = 1) #se dropea la ultima columna que representa la dificultad del input



    data[41] = data[41]=='normal'
    data[41] = data[41].replace(True, 1)





    data[41] = data[41].replace(0, -1)




    X = data.loc[:, data.columns != 41]
    y = data.loc[:, data.columns == 41]

    X = np.array(X)

    
    normalized_X = scale_features(X)
    normalized_X = pd.DataFrame(normalized_X)
    print(type(normalized_X))
    output_data = normalized_X
    output_data = output_data.join(y)






    output_data.to_csv(path_or_buf = 'Data/train.txt', index = False, mode = 'w+')







