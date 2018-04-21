import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot
from matplotlib import cm
from itertools import product
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def sigmoid(w,x):
    """ Calculate the sigmoid value
    w should be a 1,2 matrix, and x an 2,1 matrix """
    return np.asscalar(1/(1+np.exp(-np.dot(w.transpose(),x))))

def l_simple(w):
    """ Loss function, calculate the error """
    W=w
    return (sigmoid(W, np.array([[1],[0]]))-1)**2 + sigmoid(W, np.array([[0],[1]]))**2 + sigmoid(W, np.array([[1],[1]])-1)**2

def task_1():
    """ Plot the graph for task 1"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    w1 = w2 = np.arange(-6,6.1,0.1)
    X,Y = np.meshgrid(w1,w2)
    Z = np.zeros((len(X),len(Y)))
    for i, (x,y) in enumerate(product(w1,w2)):
        Z[np.unravel_index(i, (len(X),len(Y)))] = l_simple(np.array([[x],[y]]))
 
    Z = Z.transpose()#This is because we of the order we generate the elements.
    ax.view_init(18, -60) #To show the correct perspective of the figure.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("L(x)") 
    #colormap = cm._reverser.coolwarm
    surf = ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('coolwarm'))
    fig.colorbar(surf)
    pyplot.title("Task 1")
    pyplot.show()

def l_simple_t3(w):
    """ Loss function, calculate the error """
    W=np.array(w)
    return (sigmoid(W, np.array([[1],[0]]))-1)**2 + (sigmoid(W, np.array([[0],[1]])))**2 + (sigmoid(W, np.array([[1],[1]])-1)**2)

def task3_gradientDecent(lrate, itterations=1000, w=[0, 0]): 
    """ Run some itterations and return w, need to pass some w to begin with"""
    weights = w.copy()
    for x in range(itterations):
        weights[0] = weights[0] - lrate * task3_Lsimplew1(weights) #move against the gradient
        weights[1] = weights[1] - lrate * task3_Lsimplew2(weights) #move against the gradient
    print("lrate={} \t itterations={} w={} l_loss {} -> {} end".format(lrate,(x+1),weights,l_simple_t3(w),l_simple_t3(weights) ))
    return weights

def task3_Lsimplew1(w):
    """ Derivate for w1 """
    gradient_w1 = -2*np.exp(-2*w[0])/np.power((1+np.exp(-w[0])), 3) - 2*np.exp(-2*(w[0]+w[1]))/np.power(1+np.exp(-(w[0]+w[1])), 3)
    return gradient_w1

def task3_Lsimplew2(w):
    """ Derivate for w2 """
    gradient_w2 = 2*np.exp(-w[1])/np.power(1+np.exp(-w[1]), 3) - 2*np.exp(-2 * (w[0]+w[1]) ) / np.power(1 + np.exp(-(w[0]+w[1])), 3)
    return gradient_w2

def task3_showResults():
    """ Test different values of the learning rate. """

    for x in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
        task3_gradientDecent(x, 1000, w=[0,0])


def task3_find_weights():
    #calcualte the different weights from using differnt learning rates.
    lrates = [0.0001,0.001,0.01] #[ 0.00001*10**x for x in range(1,5,1)]
    lrates.extend(list(np.arange(0.1,1.1,0.1)))
    lrates.extend(range(10,110,10))
    Weights = []
    for i in range(1,4,1):
        W = []
        for learning_rate in lrates:
            W.append(task3_gradientDecent(learning_rate,10**i))
        Weights.append(W)
    return Weights, lrates, [10,100,1000]

def task3_plot_result():
    """ Plot the graph for task 3"""
    #We need to plot L() simple, and learning rate. Itterations and initial weights stay the same.

    W, lrates, iterations = task3_find_weights()
    LsimpleAll = []
    fig, ax = pyplot.subplots()
    for i, itt_w in enumerate(W):
        #We got one list of multiple weights per itteration n.
        Lsimple = []
        for weight in itt_w:
            #each element cointain w1 and w2 weights.
            Lsimple.append(l_simple_t3(weight))
        LsimpleAll.append(Lsimple)
        ax.plot(lrates,Lsimple,label=iterations[i])
    legend = ax.legend(loc='upper right')
    pyplot.ylabel("Lsimple(w)")
    pyplot.xlabel("learning rate (log form)")
    pyplot.xscale('log')
    #pyplot.yscale('logit')
    pyplot.grid()
    pyplot.title("w0=[0, 0] with different itterations")
    #pyplot.plot(lrates,Lsimple)
    pyplot.show()
 
#TASK 1
#task_1()

#TASK 3
#task3_showResults()
#Task 3 Plot
task3_plot_result() 

