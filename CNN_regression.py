#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:17:58 2022

@author: parth
"""

import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/home/parth/Desktop/uni_stuff/sensor_fusion/project/final_csv_files/cam_lidar_xyz_20Hz.csv')
df.columns = ['time','camera_x','camera_y','camera_z','lidar_x','lidar_y','lidar_z']
gt = pd.read_csv('/home/parth/Desktop/uni_stuff/sensor_fusion/project/final_csv_files/hero_xyz_20Hz.csv')
gt.columns = ['time','x','y','z']

X = np.asarray(df[['camera_x', 'camera_y', 'camera_z', 'lidar_x', 'lidar_y', 'lidar_z']])
y = np.asarray(gt[['x', 'y', 'z']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)

model = Sequential()

model.add(Dense(128, input_dim=6, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='linear'))
model.add(Dropout(0.025))
model.add(Dense(3, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, validation_split=0.2, batch_size=30,epochs=80)


from matplotlib import pyplot as plt 

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')

predictions = model.predict(X_test)

predictions_all = model.predict(X)

lr = pickle.load(open('lr_model.sav', 'rb'))

predictions_all_lr = lr.predict(X)

plt.figure(figsize=(8,5))
rn = np.arange(0,len(df+1))
plt.scatter(rn,gt['x'],s=5, label = 'Ground Truth X')
plt.scatter(rn,predictions_all_lr[:,0],s=1, label = 'Linear Regression')
# plt.scatter(x_hat[0,:874], x_hat[2,:874],s=5)
plt.ylabel('X (m)')
plt.xlabel('Time steps')
plt.title('Ground Truth vs Linear Regression in X')
plt.legend()
plt.grid()

plt.figure(figsize=(8,5))
rn = np.arange(0,len(df+1))
plt.scatter(rn,gt['y'],s=5, label = 'Ground Truth Y')
plt.scatter(rn,predictions_all[:,1],s=5, label = 'Linear Regression')
# plt.scatter(x_hat[0,:874], x_hat[2,:874],s=5)
plt.ylabel('Y (m)')
plt.xlabel('Time steps')
plt.title('Ground Truth vs Linear Regression in Y')
plt.legend()
plt.grid()

plt.figure(figsize=(8,5))
plt.scatter(gt['x'],gt['y'],s=20, label = 'Ground Truth', marker='*')
plt.plot(predictions_all[:,0],predictions_all[:,1],label = 'Linear Regression', linewidth=5, c='red', ls='dotted')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.title('Ground Truth vs Linear Regression (X,Y))')
plt.legend()
plt.grid()

plt.figure(figsize=(8,5))
plt.plot(gt['x'],gt['y'], c='blue' ,linewidth=8,label = 'Ground Truth', marker='*')
plt.plot(predictions_all_lr[:,0],predictions_all_lr[:,1],label = 'Linear Regression', linewidth=4, c='cyan', ls='dotted')
plt.plot(predictions_all[:,0],predictions_all[:,1],label = 'Neural Network Regressor', linewidth=4, c='m', ls='dotted')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.title('Ground Truth vs Neural Network Regressor vs Linear Regression (X,Y))')
plt.legend()
plt.grid()



## Predict on different Data 

# gt = pd.read_csv('final_csv_files/hero_xyz1.csv')
# gt.columns = ['time','x','y','z']
# df = pd.read_csv('final_csv_files/camera_lidar_xyz1.csv')
# df.columns = ['time','cam_x','cam_y','cam_z', 'lid_x', 'lid_y', 'lid_z']

# X = np.asarray(df[['cam_x', 'cam_y', 'cam_z', 'lid_x', 'lid_y', 'lid_z']])
# Y = np.asarray(gt[['x', 'y', 'z']])

# cnn_pred = model.predict(X)
