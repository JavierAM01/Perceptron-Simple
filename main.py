import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

from perceptron import Perceptron


f = lambda X : 3*X[0] + 2*X[1] - 2
recta = lambda x : 1 - 3/2*x
classify = lambda X : 1 if f(X) > 0 else 0


def ask_number(t=int):
    try:
        n = t(input(" > "))
        return n
    except:
        print("Invalid argument! Try again:")
        ask_number()

def update(i, X, Y, perceptron, ax, points1, points2):

    n = len(X)
    epoch = 1 + i//n
    t = i % n

    p = lambda x : -(perceptron.w[0] * x + perceptron.w0) / perceptron.w[1]

    ax.clear()
    ax.set_title("Epoch " + str(epoch) + " : Point " + str(t))
    ax.plot(points1[:,0], points1[:,1], "b.")
    ax.plot(points2[:,0], points2[:,1], "r.")
    ax.plot([-6,6], [p(-6),p(6)], "k-")  # recta del perceptron
    ax.plot([-6,6], [recta(-6),recta(6)], "k-", alpha=0.4)  # recta objetivo
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    
    x, y = X[t], Y[t]                  
    perceptron.f(x, y)

def train_perceptron():

    # ask for parameters
    print("Number of points:")
    n = ask_number()
    print("Number of epochs:")
    epochs = ask_number()
    print("Learning rate:")
    lr = ask_number(float)

    # points 
    px = 10*np.random.rand(n) - 5
    py = 10*np.random.rand(n) - 5

    points = list(zip(px, py))
    points1 = np.array([p for p in points if f(p) > 0])
    points2 = np.array([p for p in points if f(p) <= 0])

    # training data
    X = np.array(points)
    Y = np.array([classify(p) for p in points])

    perceptron = Perceptron(n_inputs=2, lr=lr)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    ani = FuncAnimation(fig, update, frames=range(epochs*n), fargs=[X, Y, perceptron, ax, points1, points2], interval=250)
    ani.save("images/ani.gif")
    plt.show()
       




if __name__ == "__main__":
    train_perceptron()