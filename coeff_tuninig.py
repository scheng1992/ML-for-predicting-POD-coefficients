# -*- coding: utf-8 -*-
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

import time


#chcek
# check scikit-learn version
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize
#input



#load learning data
field_data = np.loadtxt('....txt')
input_data = np.loadtxt('....txt')
input_data = input_data[:,[1,2,3,4]] #chosing parameters as ML input
#output
coeff = np.loadtxt('...txt')

##############################################################
field_extra = np.loadtxt('....txt') #loading validation dataset

coeff_extra = np.loadtxt('.....txt')

input_extra = np.loadtxt('....txt')

input_extra = input_extra[:3000,[1,2,3,4]]

##############################################################
#shuffle the input dataset for trainning
c = np.c_[input_data.reshape(len(input_data), -1), 
          coeff.reshape(len(coeff), -1), field_data.reshape(len(field_data), -1)]


np.random.shuffle(c)
input_data = c[:, :input_data.size//len(input_data)].reshape(input_data.shape)
# coeff  = c[:, input_data.size://
#            len(input_data):].reshape(coeff.shape)

coeff = c[:,4:154]

field_data = c[:,154:]
#load POD coefficients 
eigenvalue = np.loadtxt('.....txt')

########################################################################
#seperate the training and test dataset
train_input = input_data [:15000]
train_output_all = coeff [:15000]
test_input = input_data [15000:]
true_test_output_all = coeff [15000:]
original_test_field = field_data[15000:]

error_KNN_all = []
error_DT_all = []
std_KNN_all = []
std_DT_all = []
error_linear_all = []
error_pod_all = []
KNN_time = []
linear_time = []
DT_time = []
##############################################
# hyperparameter tuning
for index in range(2,20):    
    
    print(index)
    dim =40
    
    train_output = train_output_all[:,:dim]
    
    true_test_output = true_test_output_all[:,:dim]
    
    test_output = np.zeros((1,dim))
    ###########################################################
    t = time.time()
    
    linear_model = LinearRegression()
    
    linear_model.fit(train_input, train_output)
    
    linear_error = []
    
    for i in range(test_input.shape[0]):
        
        test_line = test_input[i,:]
        
        test_line.shape = (1,test_line.size)
        
        predict_line = linear_model.predict(test_line)
        
        test_output = np.concatenate((test_output, predict_line), axis=0)
        
    
    test_output_linear = test_output[1:,:]
    
    linear_time.append(time.time()-t)
    # plt.plot(true_test_output[:,0],'r',label = "true")
    # plt.plot(test_output[:,0],label = "model")
    # plt.title("1st coeff linear")
    # plt.legend()
    # plt.show()
    
    
    # plt.plot(true_test_output[:,1],'r',label = "true")
    # plt.plot(test_output[:,1],label = "model")
    # plt.title("2st coeff linear")
    # plt.legend()
    # plt.show()
    
    # plt.plot(true_test_output[:,2],'r',label = "true")
    # plt.plot(test_output[:,2],label = "model")
    # plt.title("3st coeff linear")
    # plt.legend()
    # plt.show()
    
    
    # plt.plot(true_test_output[:,3],'r',label = "true")
    # plt.plot(test_output[:,3],label = "model")
    # plt.title("4st coeff linear")
    # plt.legend()
    # plt.show()
    
    
    # plt.plot(true_test_output[:,4],'r',label = "true")
    # plt.plot(test_output[:,4],label = "model")
    # plt.title("5st coeff linear")
    # plt.legend()
    # plt.show()
    
    # for i in range(150):
        
    #     linear_error.append(np.linalg.norm(test_output[:,i]-true_test_output[:,i]))
    
    # linear_weighted_error = np.sum(np.array(linear_error)*eigenvalue)
    
    # print('linear_weighted_error',linear_weighted_error)
    
    
    #####################################################################
    test_output = np.zeros((1,dim))
    
    t = time.time()
    
    KN_model = KNeighborsRegressor(n_neighbors=index)
    
    KN_model.fit(train_input, train_output)
    
    KN_error = []
    
    for i in range(test_input.shape[0]):
        
        test_line = test_input[i,:]
        
        test_line.shape = (1,test_line.size)
        
        predict_line = KN_model.predict(test_line)
        
        test_output = np.concatenate((test_output, predict_line), axis=0)
        
    
    test_output_KNN = test_output[1:,:]
    
    KNN_time.append(time.time()-t)
    
    # for i in range(dim):
        
    #     KN_error.append(np.linalg.norm(test_output[:,i]-true_test_output[:,i]))
    
    # KN_weighted_error = np.sum(np.array(KN_error)*eigenvalue)
    
    # print('KN_weighted_error',KN_weighted_error)
    
    ###################################################################
    test_output = np.zeros((1,dim))
    
    t = time.time()
    
    DT_model = DecisionTreeRegressor(min_samples_split=index)
    
    DT_error = []
    
    DT_model.fit(train_input, train_output)
    
    
    for i in range(test_input.shape[0]):
        
        test_line = test_input[i,:]
        
        test_line.shape = (1,test_line.size)
        
        predict_line = DT_model.predict(test_line)
        
        test_output = np.concatenate((test_output, predict_line), axis=0)
        
    
    test_output_DT = test_output[1:,:]
    
    
    DT_time.append(time.time()-t)
    
    
    
    ##################################################################
    
    opfile2 = "data/powerIAEA18480basis.txt"
    qbasist = np.loadtxt(opfile2)
    qbasis = qbasist.T
    
    qbasis = qbasis[:,:dim]
    
    opfile1 = "data/powerIAEA18480coef.txt"
    alpha = np.loadtxt(opfile1)
    
    alpha = alpha[:,:dim]
    
    field2pod = np.dot(qbasis, true_test_output.T)
    
    field2tree_linear = np.dot(qbasis, test_output_linear.T)
    
    field2tree_KNN = np.dot(qbasis, test_output_KNN.T)
    
    field2tree_DT = np.dot(qbasis, test_output_DT.T)
    

    error_KNN = np.abs(field2tree_KNN - original_test_field.T)
    
    error_DT = np.abs(field2tree_DT - original_test_field.T)
    
    error_linear = np.abs(field2tree_linear - original_test_field.T)
    
    error_pod = np.abs(field2pod - original_test_field.T)
    
    mean_error_KNN = (np.mean(np.mean(error_KNN,axis = 0)/np.mean(original_test_field.T,axis = 0)))
    
    mean_error_DT = (np.mean(np.mean(error_DT,axis = 0)/np.mean(original_test_field.T,axis = 0)))
    
    mean_error_linear = (np.mean(np.mean(error_linear,axis = 0)/np.mean(original_test_field.T,axis = 0)))
            
    mean_error_pod = (np.mean(np.mean(error_pod,axis = 0)/np.mean(original_test_field.T,axis = 0)))
    
    std_error_KNN = (np.mean(np.std(error_KNN,axis = 0)/np.mean(original_test_field.T,axis = 0)))
    
    std_error_DT = (np.mean(np.std(error_DT,axis = 0)/np.mean(original_test_field.T,axis = 0)))
     
    error_KNN_all.append(mean_error_KNN)
    
    error_DT_all.append(mean_error_DT)
    
    std_KNN_all.append(std_error_KNN)
    
    std_DT_all.append(std_error_DT)
    
    error_linear_all.append(mean_error_linear)
    
    error_pod_all.append(mean_error_pod)


error_DT_all = np.loadtxt('result/error_DT_coeff.txt')[:18]

error_KNN_all = np.loadtxt('result/error_KNN_coeff.txt')[:18]

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 3))
       
plt.plot( list(range(2,20)), np.array(error_KNN_all)*100,linewidth=3, label = 'error KNN')


plt.plot( list(range(2,20)), np.array(std_KNN_all)*100, 'b--',linewidth = 3,label = 'std KNN')

plt.xlabel("n_neighbour", fontsize = 16)
plt.xticks(range(2,20))
plt.ylabel("error in %", fontsize = 20)
plt.legend()
plt.savefig("figures/error_coeff_KNN_2.eps",fmt ='.eps',bbox_inches='tight')
plt.show()



from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 3))

plt.plot( list(range(2,20)),np.array(error_DT_all)*100,'g', linewidth = 3, label = 'error DT')
plt.plot( list(range(2,20)),np.array(std_DT_all)*100,'g--',linewidth = 3, label = 'std DT')

plt.xlabel("samples_split", fontsize = 16)
plt.xticks(range(2,20))
plt.ylabel("error in %", fontsize = 20)
plt.legend()
plt.savefig("figures/error_coeff_DT_2.eps",fmt ='.eps',bbox_inches='tight')
plt.show()


from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 3))
plt.plot(list(range(2,20)),KNN_time, linewidth = 3, label = 'KNN_time')

#plt.plot(list(range(step+1,150,step)), linear_time, 'y', label = 'linear_time')

plt.xlabel("n_neighbour", fontsize = 16)
plt.ylabel("time(s)", fontsize = 20)
plt.xticks(range(2,20))
plt.legend()
#plt.savefig("figures/error_time_KNNceoff.eps",bbox_inches='tight')
plt.show()



figure(num=None, figsize=(8, 3))
plt.plot(list(range(2,20)), DT_time,'g', linewidth = 3, label = 'DT_time')

plt.xlabel("samples_split", fontsize = 16)
plt.ylabel("time(s)", fontsize = 20)
plt.xticks(range(2,20))
plt.legend()
#plt.savefig("figures/error_time_DTceoff.eps",bbox_inches='tight')
plt.show()


