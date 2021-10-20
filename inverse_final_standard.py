# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:52:33 2021

@author: siboc
"""
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
#from sympy import Symbol, nsolve, solve
#from sympy.solvers import solve
import pickle
import matplotlib.pyplot as plt
# check scikit-learn version
import time
# check scikit-learn version
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize

from smt.sampling_methods import LHS

SensorofQuarter = np.loadtxt('data/SensorofQuarter.txt')

KNN_model = pickle.load(open('KNN_model.sav', 'rb'))

DT_model = pickle.load(open('DT_model.sav', 'rb'))

input_data = np.loadtxt('data/inpower18480.txt')

input_data = input_data[:,[1,2,3,4]]

field_data1 = np.loadtxt('data/powerIAEA10000.txt')

field_data2 = np.loadtxt('data/powerIAEA8480.txt')

field_data = np.concatenate((field_data1,field_data2),axis = 0)



opfile2 = "data/powerIAEA18480basis.txt"
qbasist = np.loadtxt(opfile2)
qbasis = qbasist.T

qbasis = qbasis[:,:40]

opfile1 = "data/powerIAEA18480coef.txt"
alpha = np.loadtxt(opfile1)


predict_KNN = KNN_model.predict(input_data[0,:].reshape(1,4))

fieldKNN = np.dot(qbasis, predict_KNN.T)



row = 13326

input_initial = np.array([80, 1500 ,72.15, 291.01])

input_true = np.array([130, 1500 ,52.15, 291.01])

xlimits = np.array([[input_initial[0]-100, input_initial[0]+100], [input_initial[1], input_initial[1]],[input_initial[2]-30, input_initial[2]+30],
                    [input_initial[3], input_initial[3]]])

t = time.time()

sampling = LHS(xlimits=xlimits)

num = 150

group1_sample = sampling(num)

input_error_KNN = []

field_error_KNN = []

field_error_KNN_reduced = []

for i in range(group1_sample.shape[0]):
    
    test_line = group1_sample[i,:]
    
#    input_error_KNN.append(np.linalg.norm(group1_sample[i,:]-input_data[row,:]))
    
    input_error_KNN.append(np.linalg.norm((group1_sample[i,:]-input_data[row,:])/input_initial))
    
    test_line.shape = (1,test_line.size)
    
    predict_KNN = KNN_model.predict(test_line)
    
    fieldKNN= np.dot(qbasis, predict_KNN.T)
    
    field_error_KNN.append(np.linalg.norm(field_data[row,:]-fieldKNN.ravel()))
    
    field_error_KNN_reduced.append(np.linalg.norm(np.dot(SensorofQuarter,field_data[row,:].reshape(field_data[row,:].size,1))
                              -np.dot(SensorofQuarter,fieldKNN.reshape(fieldKNN.size,1))))
 
 
print ('KNN time',time.time()-t)

s = list(field_error_KNN_reduced)

#s = list(input_error_KNN)

input_error_KNN = np.array(input_error_KNN)

field_error_KNN = np.array(field_error_KNN)

field_error_KNN_reduced = np.array(field_error_KNN_reduced)

input_error_KNN = input_error_KNN[sorted(range(len(s)), key=lambda k: s[k])]

field_error_KNN = field_error_KNN_reduced[sorted(range(len(s)), key=lambda k: s[k])]


group1_sample = group1_sample[sorted(range(len(s)), key=lambda k: s[k]),:]

print('output KNN',group1_sample[0,:])


predict_KNN = KNN_model.predict(group1_sample[0,:].reshape(1,4))

fieldKNN= np.dot(qbasis, predict_KNN.T)

initial_KNN = KNN_model.predict(input_initial.reshape(1,4))

fieldinitial_KNN = np.dot(qbasis, initial_KNN.T)

#np.savetxt('inverse_fields/fieldKNN_final'+str(row)+'.txt',fieldKNN)

#np.savetxt('inverse_fields/fieldKNN_initial'+str(row)+'.txt',fieldinitial_KNN)

field_true = field_data[row,:].reshape(field_data[row,:].size,1)

field_error_KNN = field_error_KNN/np.linalg.norm(field_true)

KNN_sample = np.copy(group1_sample)

print('KNN 1', group1_sample[0,:])
print('KNN 2', group1_sample[1,:])
print('KNN 3', group1_sample[2,:])
print('KNN 4', group1_sample[3,:])
print('KNN 5', group1_sample[4,:])

print('KNN average', np.mean(group1_sample[:5,:],axis = 0))

KNN_parm_mean = np.mean(group1_sample[:5,:],axis = 0)

predict_KNN = KNN_model.predict(group1_sample[0,:].reshape(1,4))

fieldKNN= np.dot(qbasis, predict_KNN.T)

plt.plot(KNN_sample[:,0],KNN_sample[:,2],'yo',alpha = 0.5)

plt.plot(KNN_sample[:5,0],KNN_sample[:5,2],'bo')

plt.plot(KNN_sample[0,0],KNN_sample[0,2],'mo')

plt.plot(input_initial[0],input_initial[2],'ko')

plt.plot(input_true[0],input_true[2],'ro')

plt.plot(KNN_parm_mean[0],KNN_parm_mean[2],'go')
plt.xlabel('$\mu^1$',fontsize = 16)
plt.ylabel('$\mu^3$',fontsize = 16)
plt.savefig("figures/KNN_c5.eps",fmt = '.eps')

plt.show()


###################################################################################

row = 13326

input_initial = np.array([80, 1500 ,72.15, 291.01])

input_true = np.array([130, 1500 ,52.15, 291.01])


xlimits = np.array([[input_initial[0]-100, input_initial[0]+100], [input_initial[1], input_initial[1]],[input_initial[2]-30, input_initial[2]+30],
                    [input_initial[3], input_initial[3]]])

num = 100
t = time.time()

sampling = LHS(xlimits=xlimits)


group1_sample = sampling(num)

input_error_DT = []

field_error_DT = []

field_error_DT_reduced = []

for i in range(group1_sample.shape[0]):
    
    test_line = group1_sample[i,:]
    
    #input_error_DT.append(np.linalg.norm(group1_sample[i,:]-input_data[row,:]))
    
    input_error_DT.append(np.linalg.norm((group1_sample[i,:]
                                         -input_data[row,:])/input_initial))
    
    test_line.shape = (1,test_line.size)
    
    predict_DT = DT_model.predict(test_line)
    
    fieldDT= np.dot(qbasis, predict_DT.T)
    
    field_error_DT.append(np.linalg.norm(field_data[row,:]-fieldDT.ravel()))
    

    field_error_DT_reduced.append(np.linalg.norm(np.dot(SensorofQuarter,field_data[row,:].reshape(field_data[row,:].size,1))
                                  -np.dot(SensorofQuarter,fieldDT.reshape(fieldDT.size,1))))
    
    

print ('DT time',time.time()-t)


s = list(field_error_DT_reduced)

#s = list(input_error_DT)

input_error_DT = np.array(input_error_DT)

field_error_DT = np.array(field_error_DT)

field_error_DT_reduced = np.array(field_error_DT_reduced)

input_error_DT = input_error_DT[sorted(range(len(s)), key=lambda k: s[k])]

field_error_DT = field_error_DT_reduced[sorted(range(len(s)), key=lambda k: s[k])]


group1_sample = group1_sample[sorted(range(len(s)), key=lambda k: s[k]),:]

print('output DT',group1_sample[0,:])

predict_DT = DT_model.predict(group1_sample[0,:].reshape(1,4))

fieldDT= np.dot(qbasis, predict_DT.T)

initial_DT= DT_model.predict(input_initial.reshape(1,4))

fieldinitial_DT = np.dot(qbasis, initial_DT.T)



#print('DT u', group1_sample[0,:])

DT_sample = np.copy(group1_sample)

print('DT 1', group1_sample[0,:])
print('DT 2', group1_sample[1,:])
print('DT 3', group1_sample[2,:])
print('DT 4', group1_sample[3,:])
print('DT 5', group1_sample[4,:])

print('DT average', np.mean(group1_sample[:5,:],axis = 0))

DT_parm_mean = np.mean(group1_sample[:5,:],axis = 0)

###########################################################################

predict_DT_initial = DT_model.predict(input_initial.reshape(1,4))

fieldDT_initial= np.dot(qbasis, predict_DT_initial.T)


predict_KNN_initial = KNN_model.predict(input_initial.reshape(1,4))

fieldKNN_initial= np.dot(qbasis, predict_KNN_initial.T)

field_true = field_data[row,:].reshape(field_data[row,:].size,1)


predict_DT_mean = DT_model.predict(DT_parm_mean.reshape(1,4))

fieldDT_mean= np.dot(qbasis, predict_DT_mean.T)

predict_DT = DT_model.predict(group1_sample[0,:].reshape(1,4))

fieldDT= np.dot(qbasis, predict_DT.T)


predict_KNN_mean = KNN_model.predict(KNN_parm_mean.reshape(1,4))

fieldKNN_mean = np.dot(qbasis, predict_KNN_mean.T)

predict_KNN_mean = KNN_model.predict(KNN_parm_mean.reshape(1,4))

fieldKNN_mean = np.dot(qbasis, predict_KNN_mean.T)

print ('input_initial',input_initial)
print('field_true-predict_KNN_initial',np.linalg.norm(field_true-
                                                     fieldKNN_initial)/np.linalg.norm(field_true))

print('field_true-fieldKNN_mean',np.linalg.norm(field_true-
                                                     fieldKNN_mean)/np.linalg.norm(field_true))

print('field_true-predictKNN',np.linalg.norm(field_true-
                                                     fieldKNN)/np.linalg.norm(field_true))

print('field_true-fieldDT',np.linalg.norm(field_true-
                                                     fieldDT)/np.linalg.norm(field_true))


print('field_true-predict_DT_initial',np.linalg.norm(field_true-
                                                     fieldDT_initial)/np.linalg.norm(field_true))

print('field_true-fieldDT_mean',np.linalg.norm(field_true-
                                                     fieldDT_mean)/np.linalg.norm(field_true))


plt.plot(DT_sample[:,0],DT_sample[:,2],'yo',alpha = 0.5)

plt.plot(DT_sample[:5,0],DT_sample[:5,2],'bo')

plt.plot(DT_sample[0,0],DT_sample[0,2],'mo')

plt.plot(input_initial[0],input_initial[2],'ko')

plt.plot(input_true[0],input_true[2],'ro')

plt.plot(DT_parm_mean[0],KNN_parm_mean[2],'go')

plt.xlabel('$\mu^1$',fontsize = 16)
plt.ylabel('$\mu^3$',fontsize = 16)
plt.savefig("figures/DT_c5.eps",fmt = '.eps')
plt.show()



