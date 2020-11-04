# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:35:08 2018

@author: AFUMETTO
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv", sep = ";")

data.layer_height = data.layer_height*100
data.elongation = data.elongation*100
#%%
data.material = [0 if each == "abs" else 1 for each in data.material]
# abs = 0, pla = 1

data.infill_pattern = [0 if each == "grid" else 1 for each in data.infill_pattern]
# grid = 0, honeycomb = 1


y = data.material.values
x_data = data.drop(["material"],axis=1)

#%%
#seperate dataframe
"""
rou = df.iloc[:, 9:10].values
ten = df.iloc[:, 10:11].values
elo = df.iloc[:, 11:12].values
x = df.iloc[:, 0:9].values
y = df.iloc[:, 9:12].values

y3 = rou.reshape(50,)
y4 = ten.reshape(50,)
y5 = elo.reshape(50,)
"""
absm = data[data.material == 0]
pla = data[data.material == 1]

plt.scatter(absm.fan_speed,absm.tension_strenght,color="red",label="ABS",alpha= 0.5)
plt.scatter(pla.fan_speed,pla.tension_strenght,color="green",label="PLA",alpha= 0.5)
plt.xlabel("Fan Hızı")
plt.ylabel("Çekme Dayanımı")
plt.legend()
plt.show()

plt.scatter(absm.layer_height,absm.roughness,color="blue",label="ABS",alpha= 0.9)
plt.scatter(pla.layer_height,pla.roughness,color="pink",label="PLA",alpha= 0.9)
plt.xlabel("Katman Yüksekliği")
plt.ylabel("Yüzey Pürüzlülüğü")
plt.legend()
plt.show()

plt.scatter(absm.layer_height,absm.tension_strenght,color="orange",label="ABS",alpha= 0.5)
plt.scatter(pla.layer_height,pla.tension_strenght,color="brown",label="PLA",alpha= 0.5)
plt.xlabel("Katman Yüksekliği")
plt.ylabel("Çekme Dayanımı")
plt.legend()
plt.show()


plt.scatter(pla.layer_height,pla.tension_strenght,color="green",label="PLA",alpha= 0.5)
plt.xlabel("Katman Yüksekliği")
plt.ylabel("Çekme Dayanımı")
plt.legend()
plt.show()

plt.scatter(absm.layer_height,absm.tension_strenght,color="red",label="ABS",alpha= 0.5)
plt.xlabel("Katman Yüksekliği")
plt.ylabel("Çekme Dayanımı")
plt.legend()
plt.show()

#%%
# normalization 
x_norm = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm,y,test_size = 0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    print(" {} nn score: {} ".format(each,knn2.score(x_test,y_test)))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Dense(32,input_dim=11))
model.add(BatchNormalization(axis = -1))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_data,y, epochs=500, batch_size =32, validation_split= 0.20)



#%% Tahmin
a1 = 4 #layer_height*100
a2 = 5 #wall_thickness
a3 = 60 #infill_density
a4 = 0 #infilkk_pattern
a5 = 232 #nozzle_temperature 
a6 = 74 #bed_temperature
a7 = 90 #print_speed
a8 = 100 #fan_speed
a9 = 150 #roughness
a10 = 30 #tension_strenght
a11 = 200 #elangation*100

tahmin = np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]).reshape(1,11)
print(model.predict_classes(tahmin))

if model.predict_classes(tahmin) == 0: 
    print("Kullanılan malzeme ABS'dir.")
else:   
    print("Kullanılan malzeme PLA'dır.")
    
#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data.infill_density
y = data.wall_thickness
z = data.tension_strenght



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Doluluk Oranı')
ax.set_ylabel('Duvar Kalınlığı')
ax.set_zlabel('Çekme Dayanımı')

plt.show()