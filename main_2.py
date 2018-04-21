import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import time

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def logistic_z(z): 
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x): 
    return logistic_z(np.inner(w,x))

#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features

def batch_GD(data, learning_rate=0.1, niterations=1000, randomW=True):
    #x_train and y_train are both np.arrays.
    dim = data.shape[1]-1 #How many weights we got(number of colums - 1)
    num_n = data.shape[0] #How many training examples.
    if(randomW):
        W = np.reshape(np.random.rand(dim),(dim,1)).transpose() #Initalize weights.
    else:
        W = np.reshape(np.zeros(dim),(dim,1)).transpose()
    #Split training data from value.
    x_train = np.delete(data, dim,axis=1)
    y_train = np.reshape(data[:,dim],(num_n,1))
    weights_all.append(W)
    for t in range(niterations):
        #need to update the gradient after every itteration.
        grad = np.reshape(np.zeros(dim),(dim,1)).transpose()
        for d in range(num_n):#for every data set
            wTx = np.dot(W, x_train[d])
            A = np.exp(-wTx)
            B = np.power(1+A,2)
            E = logistic_z(wTx) - y_train[d]
            grad += E*A/B*x_train[d]
        W = W - (learning_rate/num_n)*grad
        weights_all.append(W)
    return W

def stochastic_GD(data, learning_rate=0.1, niterations=1000,randomW=True):
    #x_train and y_train are both np.arrays.
    dim = data.shape[1] #How many weights we got(number of colums - 1)
    num_n = data.shape[0] #How many training examples.
    if(randomW):
        W = np.reshape(np.random.rand(dim-1),(dim-1,1)).transpose() #Initalize weights.
    else:
        W = np.reshape(np.zeros(dim-1),(dim-1,1)).transpose() #Initalize weights.
    #We need to shuffle the data set, so that we get a random value for each itteration(exluding the ones previouse used)
    np.random.shuffle(data)
    #Split training data from value.
    x_train = np.delete(data, dim-1,axis=1)
    y_train = np.reshape(data[:,dim-1],(num_n,1))
    weights_all.append(W)
    for t in range(niterations):#Loop tru n iterations.

        #for d in range(num_n):#loop tru ever data point.
        wTx = np.dot(W, x_train[t])
        A = np.exp(-wTx)
        B = np.power(1+A,2)
        E = logistic_z(wTx) - y_train[t]
        grad = E*A/B*x_train[t]
        W = W - learning_rate*grad
        weights_all.append(W)
    return W


def loadData(file):
    df = pd.read_csv(file, sep='\t')
    data = df.values
    #Augment the datafiles, to add bias term.
    dim = data.shape[0]
    ones = np.ones(dim)
    ones_2d = np.reshape(ones,(dim,1))
    data = np.hstack((ones_2d,data))
    return data

def classify_2(w, x):
    z = np.dot(w,x)
    return 0 if (logistic_z(z)<0.5) else 1

def train_and_plot(data_train,data_test,training_method,learn_rate=0.1,niter=1000, plot_scatter = True):
    plt.figure()
    #train data

    dim = data_test.shape[1]-1
    num_n = data_test.shape[0]
    
    x_test = np.delete(data_test, dim,axis=1)
    y_test = np.reshape(data_test[:,dim],(num_n,1))
    start = time.time() # start timer
    #Train weights
    w=training_method(data_train,learn_rate,niter)
    end = time.time() # end timer
    time_elapsed = end-start #calculate difference.
    print("{} \n{} seconds".format(training_method.__name__ ,time_elapsed))
    print("weights: {}".format(np.squeeze(w)))
    #measure error with the trained weights.
    if(plot_scatter):
        error=0
        for d in range(num_n): 
            #Check the data.
            #We plot the x1,x2 with the colour and shape for correct or incorrect estimates.
            #rint(np.reshape(x_test[d],(dim,1)))
            estimate = classify_2(w,x_test[d])
            if(y_test[d] == 0):
                if(estimate != y_test[d]):
                    plt.plot(x_test[d][1],x_test[d][2], 'rx')
                    error+=1
                else:
                    plt.plot(x_test[d][1],x_test[d][2], 'ro')
            else:
                if(estimate != y_test[d]):
                    plt.plot(x_test[d][1],x_test[d][2], 'gx')
                    error+=1
                else:
                    plt.plot(x_test[d][1],x_test[d][2], 'go')
        print("{} % error".format(error/num_n*100))
    else:
        print("checking error")
        error_rates=[]
        for i,weigh in enumerate(weights_all):
            error=0
            w = weights_all[i]
            for d in range(num_n):
                estimate = classify_2(w,x_test[d])
                #print(estimate,y_test[d])
                if(y_test[d] != estimate):
                    error+=1
            #print(error/num_n*100)
            error_rates.append(error/num_n)
            
        plt.plot(range(len(error_rates)),error_rates)
        plt.ylim(0,1)
        plt.ylabel("erro_rate")
        plt.xlabel("itteration")
        title = training_method.__name__ 
        plt.title(title)    

    if(plot_scatter):
        plt.ylabel("x2")
        plt.xlabel("x1")
        title = "red=0, green=1 " + training_method.__name__
        x = np.arange(-2,4,0.1)
        plt.plot(x,b(x,np.squeeze(w)),'k-')
        plt.title(title)
        #scale figure according to data points.
        min_x0, min_x1, min_x2 = x_test.min(axis=0)
        max_x0, max_x1, max_x2 = x_test.max(axis=0)
        plt.xlim(min_x1,max_x1)
        plt.ylim(min_x2,max_x2)
        #We want to plot the error for each itteration instead.

    plt.show()

def b(x,W):
    """ function that correspond to linear boundary with our weights """
    return -W[0]/W[2] - W[1]/W[2]*x
    #return -np.squeeze(np.asarray(W[0]))/np.squeeze(np.asarray(W[2])) - (np.squeeze(np.asarray(W[1])))/(np.squeeze(np.asarray(W[2])))*x


weights_all = [] #store error rates.

data_big_nonsep_test = loadData("data/data_big_nonsep_test.csv")
data_big_nonsep_train = loadData("data/data_big_nonsep_train.csv")

data_big_separable_train = loadData("data/data_big_separable_train.csv")
data_big_separable_test = loadData("data/data_big_separable_test.csv")

data_small_nonsep_train = loadData("data/data_small_nonsep_train.csv")
data_small_nonsep_test = loadData("data/data_small_nonsep_test.csv")

data_small_separable_train = loadData("data/data_small_separable_train.csv")
data_small_separable_test = loadData("data/data_small_separable_test.csv")

#train_and_plot(data_small_separable_train, data_small_separable_test, stochastic_GD)
#train_and_plot(data_small_nonsep_train, data_small_nonsep_test, stochastic_GD, plot_scatter=False)
#train_and_plot(data_big_nonsep_train, data_big_nonsep_test, stochastic_GD, plot_scatter = False)
#train_and_plot(data_big_separable_train, data_big_separable_test, stochastic_GD, plot_scatter=False)

#train_and_plot(data_small_separable_train, data_small_separable_test, batch_GD)
#train_and_plot(data_small_nonsep_train, data_small_nonsep_test, batch_GD, plot_scatter=False)
#train_and_plot(data_big_nonsep_train, data_big_nonsep_test, batch_GD)
#train_and_plot(data_big_separable_train, data_big_separable_test, batch_GD, plot_scatter=False)