import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess_multi_pop as pm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
print(tf.config.list_physical_devices())
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, BatchNormalization, Dropout,concatenate
from tensorflow.keras.models import Model
#fonction permettant de transformer les données dans la bonne numérisation
def __data_load(data):
    columns=data.columns
    data['Age']=np.where(data['Age']!='110+',data['Age'],111)
    for col in columns:
        data[col]=np.where(data[col]!='.',data[col],9999)
        data[col]=pd.to_numeric(data[col])

    data=data[data['Age']<100]
    data=data[data['Year']>=1950]
    data=data[data['Year']<=2018]
    data.index=np.arange(data.shape[0])  ### renommer les index de 0 jusqu'à la taille de data
    return data



### Transform the based dataset to an matrix of mortality rates of male people in this case. 
#The matrix has age on row and year on columns
def __data_reshaping(data):
    data=__data_load(data)
    mat=pd.DataFrame(index=np.unique(data['Age']),columns=np.unique(data['Year']))
    n=0
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            mat.iloc[mat.index[i],mat.index[j]]=data.loc[n+i,"Male"]
        n=n+mat.shape[0]
    years=np.arange(1950,(1950+mat.shape[1]))
    ages=np.arange(0,(mat.shape[0]))

    return mat, ages, years     
