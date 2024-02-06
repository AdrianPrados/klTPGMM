import math
import sys
import os
import pickle
import time
import itertools
import random
from typing import Dict
import networkx as nx
from dtw import dtw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from scipy import linalg
from scipy.stats import multivariate_normal
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation
from sklearn.mixture import GaussianMixture
from plot3DGMM import plot_GMM
from GMR import gaussPDF, GMR
from newPoint import rotation_matrix, transform_gmm
from SimulEnv import Robot, Environment


from sClass import s
from pClass import p
from matplotlib import pyplot as plt
from TPGMM_GMR import TPGMM_GMR
from copy import deepcopy
import time
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
import scipy
from collections import OrderedDict
from scipy.stats import zscore
import os
from scipy.spatial.transform import Rotation


nbSamples = 5 #Number of data
nbVar = 4 #Number of dimensions (x,y,z,t)
nbFrames = 2
nbStates = 12 #Number of Gaussians
nbData = 150

#? Data visualization
# 3D visualization of dataset
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(projection = '3d')
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
ax1.set_xlim([-4, 8])

# x,y,z,time
data = np.zeros((1,4))


# For the data to be generated,
# x cordinate is at 0 for start and end
# y cordinate ranges from 0 to 5
# z cordinate is the amplitude that changes from 2 to 3

# same X values
#X = np.array([0]*150, dtype=float)
#X= np.linspace(0, 5, nbData)
# Generar datos para X
X_base = np.linspace(0, 5, nbData)

# Agregar variaciones utilizando una función seno
variaciones = 0.5 * np.sin(np.linspace(0, 10, nbData))  
X = X_base + variaciones
#print(X)
# Y over a range of 0 to 5
#Y = np.linspace(0, 5, nbData)
Y_base = np.linspace(0, 5, nbData)
variaciones = 0.5 * np.sin(np.linspace(0, 28, nbData))  
Y = Y_base + variaciones

# Wavelength parameters for Z
amplitudes = np.linspace(2,3,nbSamples)
#print(amplitudes)
frequency = 0.1
phase = 0.0
print(amplitudes)


# Ángulos de Euler en grados
roll = 45.0
pitch = 30.0
yaw = 60.0

# Convertir ángulos a radianes
roll_rad = np.radians(roll)
pitch_rad = np.radians(pitch)
yaw_rad = np.radians(yaw)

# Crear matriz de orientación 3x3
matriz_orientacion = Rotation.from_euler('zyx', [yaw_rad, pitch_rad, roll_rad]).as_matrix()

# Crear una matriz de transformación 4x4 con posición en el espacio
matriz = np.eye(4)
matriz[:3, :3] = matriz_orientacion  # Substituir la submatriz superior izquierda por la matriz de orientación

# Definir la posición en el espacio (por ejemplo, [1, 2, 3])
posicion = np.array([0,0,0])

# Asignar la posición a la última columna
matriz[:3, 3] = posicion


# for different amplitudes
for a in amplitudes:
    # values for Z. half wavelength of sine wave
    Z = a * np.sin(2 * np.pi * frequency * Y + phase)
    temp = np.stack((X, Y, Z), axis=1)
    # add time component
    temp = np.hstack((temp, np.array([i for i in range(len(temp))]).reshape(-1,1)))
    # append the current dataset to the data array
    data = np.vstack((data, temp/100))
    ax1.scatter(X, Y, Z, marker='.')
#print(temp)
# remove first row which is reference
dataCopy = data[1:,:]
print(dataCopy[0,:3])
# multiple copy of the same data to double the dataset length
#data = np.vstack((data, data)))
plt.show()
#time.sleep(1000)
print(matriz)
slist=[]
for i in range(nbSamples):
    pmat = np.empty(shape=(nbFrames, nbData), dtype=object)
    tempData = dataCopy[0:149].T
    #print("Len tempData: {}".format(tempData0[0]))
    for j in range(nbFrames):
        tempA = matriz
        if j == 0:
            tempB = tempData[0,:4]
        else:
            tempB = tempData[-1,:4]
        for k in range(nbData):
            pmat[j, k] = p(tempA, tempB.reshape(len(tempB), 1), np.linalg.inv(tempA), nbStates)
    slist.append(s(pmat, tempData, tempData.shape[1], nbStates))
    dataCopy[0:149] = dataCopy[150:299]

#* Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
TPGMMGMR = TPGMM_GMR(nbStates, nbFrames, nbVar)
print("Funciono hasta aqui")
print(len(slist))
# Learning the model-------------------------------------------------------------------------------------------------- #
TPGMMGMR.fit(slist)

# Reproduction for parameters used in demonstration------------------------------------------------------------------- #
rdemolist = []
mus_init=[]
sigmas_init=[]
mus=[] #? Los valores se gaurdan en [n][0][j][i] donde n es el numero de demos y i el nuemro de gaussianas y j es la columna
sigmas=[] #? los valores e gaurdan en [n][0][j][e][i] n y i son iguales; j es 0 o 1 que es la columna; e es entr 0 y 1 que es el primer o el segundo valor de la columna
for n in range(nbSamples):
    print(slist[n].Data[0:3,0])
    rdemolist.append(TPGMMGMR.reproduce(slist[n].p, slist[n].Data[0:3,0]))
    
    
#* Demos--------------------------------------------------------------------------------------------------------------- #
xaxis = 1
yaxis = 2
zaxis = 3
xlim = [-4, 4]
ylim = [-4, 4]
z_lim = [-4, 4]
fig = plt.figure()
ax1 = fig.add_subplot(141)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
#ax1.set_zlim(z_lim)
ax1.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
plt.title('Demonstrations')
for n in range(nbSamples):
    for m in range(nbFrames):
        ax1.plot([slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[xaxis,0] + slist[n].p[m,0].A[xaxis,yaxis]],
                [slist[n].p[m,0].b[yaxis,0], slist[n].p[m,0].b[yaxis,0] + slist[n].p[m,0].A[yaxis,yaxis]],
                [slist[n].p[m,0].b[zaxis,0], slist[n].p[m,0].b[zaxis,0] + slist[n].p[m,0].A[zaxis,yaxis]],
                lw=7, color=[0, 1, m])
        ax1.scatter(slist[n].p[m,0].b[xaxis,0], slist[n].p[m,0].b[yaxis,0], slist[n].p[m,0].b[zaxis,0], marker='.')
    ax1.scatter(slist[n].Data[xaxis,0], slist[n].Data[yaxis,0], slist[n].Data[zaxis,0], marker='.')
    ax1.plot(slist[n].Data[xaxis,:], slist[n].Data[yaxis,:], slist[n].Data[zaxis,:])
print("Funciono hasta aqui 2")
plt.show()
#* Reproductions with training parameters------------------------------------------------------------------------------ #
ax2 = fig.add_subplot(142)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
plt.title('Reproductions with same task parameters')
for n in range(nbSamples):
    listMu,listSig,_ = TPGMMGMR.plotReproduction(rdemolist[n], xaxis, yaxis, ax2, showGaussians=False)
    mus.append(listMu)
    sigmas.append(listSig)