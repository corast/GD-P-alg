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
    #colormap = cm._reverser.coolwarm
    surf = ax.plot_surface(X,Y,Z, cmap=cm.get_cmap('coolwarm'))
    fig.colorbar(surf)
    pyplot.title("Task 1")
    pyplot.show()

w=np.array([[1,0]])
w2=np.array([[1],[0]])
#print(w.shape)
#print(w2.shape)

#print(l_simple(w2))

task_1()