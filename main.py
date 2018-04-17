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
    #print(W)
    return (sigmoid(W, np.array([[1],[0]]))-1)**2 + (sigmoid(W, np.array([[0],[1]])))**2 + (sigmoid(W, np.array([[1],[1]])-1)**2)**2

def task_1():
    """ Plot the graph for task 1"""
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    w1 = w2 = np.arange(-6,6.1,0.1)
    X,Y = np.meshgrid(w1,w2)
    Z = np.zeros((len(X),len(Y)))
    for i, (x,y) in enumerate(product(w1,w2)):
        Z[np.unravel_index(i, (len(X),len(Y)))] = l_simple(np.array([[x],[y]]))
    
    ax.view_init(30, 145) #To show the correct perspective of the figure.
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
    return (sigmoid(W, np.array([[1],[0]]))-1)**2 + (sigmoid(W, np.array([[0],[1]])))**2 + (sigmoid(W, np.array([[1],[1]])-1)**2)**2

def task3_gradientDecent(lrate, itterations, w=[0, 6]): 
    """ Run some itterations and return w, need to pass some w to begin with"""
    weights = w.copy()
    for x in range(itterations):
        #print("epoch {} weights {} l_loss {}".format(x, weights,l_simple_t3(weights) ))
        #print("gradw1 {} change {}".format(task3_Lsimplew1(weights), lrate * task3_Lsimplew1(weights)))
        #print("gradw2 {} change {}".format(task3_Lsimplew2(weights),lrate * task3_Lsimplew2(weights)))
        weights[0] = weights[0] - lrate * task3_Lsimplew1(weights)
        weights[1] = weights[1] - lrate * task3_Lsimplew2(weights)
    print("lrate={} \t itterations={} w={} l_loss {} -> {} end".format(lrate,(x+1),weights,l_simple_t3(w),l_simple_t3(weights) ))
    return weights

def task3_Lsimplew1(w):
    """ Derivate for  """
    gradient_w1 = -2*np.exp(-2*w[0])/np.power((1+np.exp(-w[0])), 3) - 2*np.exp(-2*(w[0]+w[1]))/np.power(1+np.exp(-(w[0]+w[1])), 3)
    return gradient_w1

def task3_Lsimplew2(w):
    gradient_w2 = 2*np.exp(-w[1])/np.power(np.exp(-w[1]), 3) - 2*np.exp(-2 * (w[0]+w[1]) ) / np.power(1 + np.exp(-(w[0]+w[1])), 3)
    return gradient_w2

def task3_showResults():
    task3_gradientDecent(0.0001,100)
    task3_gradientDecent(0.001,100)
    task3_gradientDecent(0.01,100)
    task3_gradientDecent(0.1,100)
    task3_gradientDecent(1,100)
    task3_gradientDecent(10,100)
    task3_gradientDecent(100,100)

#TASK 1
#task_1()

#TASK 3
task3_showResults()