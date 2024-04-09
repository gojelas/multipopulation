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


### Generate the features and the target variables.  
"""
XX_x1_t0=(X_x1-1_t0, X_x1,t0, X_x1+1_t0), Y_x1_t0=X_x1,t0
.                                  .
.                                  .
.                                  .
XX_x1_tn=(X_x1-1_tn, X_x1,tn, X_x1+1_tn), Y_x1_tn=X_x1,tn
XX_x2_t0=(X_x2-1_t0, X_x2,t0, X_x2+1_t0), Y_x2_t0=X_x2,t0
.                                  .
.                                  .
.                                  .
XX_x2_tn=(X_x2-1_tn, X_x2,tn, X_x2+1_tn), Y_x2_tn=X_x2,tn
.                                  .
.                                  .
.                                  .

"""   
def __variables_generating(data,tau0=1):
    #sc=MinMaxScaler()
    #df1=sc.fit_transform(__data_reshaping(data)[0])
    #data=pd.DataFrame(df1,index=__data_reshaping(data)[1],columns=__data_reshaping(data)[2])
    n=2*tau0+1
    data=__data_reshaping(data)[0]
    X=pd.DataFrame(columns=['X_{}'.format(r) for r in range(n)])
    
    y=pd.DataFrame(columns=['y'])
    
    for i in range(tau0,(data.shape[0]-tau0)):
        u=data.loc[i,:].T
        u.name='y'
        y=pd.concat([y,u],axis=0)
        s=data.loc[(i-tau0):(i+tau0),:].T
        s.columns=['X_{}'.format(r) for r in range(n)]
        X=pd.concat([X,s],axis=0)
        #X.columns=['X_{}'.format(r) for r in range(n)]
        #y.columns="a_{}".format(i)
    
    return X,y
        

### Function reshapes each country data at the shape of LSTM
"""
Y_x1_t0=f((XX_x1_t0-T,XX_x1_t0-T+1,..., XX_x1_t0-1).T)
"""
def __variables_reshaping(data,tau0=1,look_back=4):
    df1=__data_reshaping(data)[0]
    T=df1.shape[1] ##longueur temporelle de data
    N=df1.shape[0]-tau0 ##longueur en âge de data
    X,y=__variables_generating(data,tau0)
    X.index=np.arange(X.shape[0])
    y.index=np.arange(y.shape[0])

    X_array=[] 
    y_array=[]  

    for n in range(N-tau0):
        x=T*n
        for i in range(T-look_back):
            X_array.append(X[(x+i):(x+i+look_back)])
            y_array.append(y.loc[(x+i+look_back),:])

    X_array, y_array=np.array(X_array),np.array(y_array)
    return X_array, y_array



### Apply the standard normalization to the train data set (below 2005)
def train_test_data(data):
    data=__data_load(data)
    
    train=data[data["Year"]<=2005]

    df_min=np.min(train["Male"])
    df_max=np.max(train["Male"])

    train["Male"]=(train["Male"]-df_min)/(df_max-df_min) ### normalize the train male data by using the min and max of this

    test=data[data["Year"]>=2005] ### recover the test data after 2005

    test["Male"]=(test["Male"]-df_min)/(df_max-df_min) ### normalize the test data using the min and max of the train set

    X_test,y_test=__variables_reshaping(test)

    X_train,y_train=__variables_reshaping(train)

    X_train=np.asarray(X_train).astype('float32') ### change the type "object" to the type "float", this is due to the use of list
    y_train=np.asarray(y_train).astype('float32')
    X_test=np.asarray(X_test).astype('float32')
    y_test=np.asarray(y_test).astype('float32')
    return X_train, y_train, X_test, y_test

    
    

import os
 
### import data set for all country
directory = '/Users/gojelastat/Desktop/Thèse/Projet 2/Données'
data={}
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and filename.endswith('.txt'):
        data[filename.split('.')[0]]=pd.read_csv(f,header=1,delimiter="\s+") ### filename.split('.')[0] split the name by . and recover the code of the country
        ### in this line, we fill the dictionary "data" with the different data of country and their code
### Classification des observations:
#from 
X_train={}
X_test={}
y_train={}
y_test={}
inputs_={}
lstm_couches_regions={}
outputs_={}

input_time_step=4
N_variables=3

for code, country in data.items():
    X_train[code], y_train[code], X_test[code], y_test[code]=train_test_data(country)
    inputs_[code]=Input(shape=(input_time_step,N_variables))
    lstm_couches_regions[code]=LSTM(128, return_sequences=False)(inputs_[code])

### concatenation des sorties LSTM
merged=concatenate([lstm_couches_regions[code] for code,country in data.items()])

### couche dense de sortie
for code, country in data.items():
    outputs_[code]=Dense(1)(merged)

### Le modèle

"""We recover the inputs layers data on the first list and the outputs layers on the second list """
model=Model(inputs=[inputs_[code] for code, country in data.items()], 
            outputs=[outputs_[code] for code, country in data.items()])

    
for code, country in data.items():
    print(code,X_train[code].shape)
### Compiler le modèle
model.compile(optimizer='adam',loss='mse')

### entrainer le modèle
# Entraîner le modèle
model.fit([X_train[code] for code,country in data.items()], 
          [y_train for code,country in data.items()], epochs=10, batch_size=32)

# Utiliser le modèle pour des prédictions
#predictions = model.predict([input1_data, input2_data])

url='/Users/gojelastat/Desktop/Thèse/Projet 2/Données/AUS.Mx_1x1.txt'
aus=pd.read_csv(url,header=1,delimiter="\s+")
#aus,a,t=__data_reshaping(aus)
url='/Users/gojelastat/Desktop/Thèse/Projet 2/Données/AUT/STATS/Mx_1x1.txt'
aut=pd.read_csv(url,header=1,delimiter="\s+")
#aut=__data_load(aut)
url='/Users/gojelastat/Desktop/Thèse/Projet 2/Données/FRATNP/STATS/Mx_1x1.txt'
fra=pd.read_csv(url,header=1,delimiter="\s+")
fra=__data_load(fra)
url='/Users/gojelastat/Desktop/Thèse/Projet 2/Données/BLR/STATS/Mx_1x1.txt'
blr=pd.read_csv(url,header=1,delimiter="\s+")
blr=__data_load(blr)
url='/Users/gojelastat/Desktop/Thèse/Projet 2/Données/CZE.Mx_1x1.txt'
grc=pd.read_csv(url,header=1,delimiter="\s+")






model=Sequential()
model.add(LSTM(units=128,input_shape=(x_aus_train.shape[1],x_aus_train.shape[2]),return_sequences
          =True))
model.add(LSTM(64,kernel_initializer='normal'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()
model.fit(x_aus_train,y_aus_train, validation_data=(x_aus_test,y_aus_test),
        epochs=10,batch_size=12)