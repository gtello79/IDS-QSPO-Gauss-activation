#!/usr/bin/env python
# coding: utf-8

# In[354]:


import numpy as np
import pandas as pd
from sklearn import preprocessing #no se si se podra usar esto para label encoder xd


if __name__ == "__main__":

# In[355]:


    DATA_PATH = 'Data/KDDTest+.txt'
    data = data = pd.read_csv(DATA_PATH, sep=",", header=None)
    a = 0.1
    b = 0.99


    # In[356]:

    encoder = preprocessing.LabelEncoder()

    #pero que eficiente xdxd
    data[1]= encoder.fit_transform(data[1]) 
    data[2]= encoder.fit_transform(data[2]) 
    data[3]= encoder.fit_transform(data[3]) 


    # In[357]:


    data = data.drop(42,axis = 1) #se dropea la ultima columna que representa la dificultad del input
    data = data.drop(19, axis = 1) #el atributo 19 por alguna razon tiene puros 0


    # In[358]:


    data[41] = data[41]=='normal'
    data[41] = data[41].replace(True, 1)


    # In[359]:


    data[41] = data[41].replace(0, -1)


    # In[360]:


    X =data.loc[:, data.columns != 41]
    y = data.loc[:, data.columns == 41]


    # In[361]:


    normalized_X=(X-X.min())/(X.max()-X.min())


    # In[374]:


    normalized_X = (b-a)*normalized_X + a #esto lo encontre un poco raro pero sale en el pdf xd
    output_data = normalized_X
    output_data = output_data.join(y)


    # In[375]:


    output_data.to_csv(path_or_buf = 'Data/test.txt', index = False, mode = 'w+')







