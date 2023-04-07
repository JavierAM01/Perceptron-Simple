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
    ax.set_title(f"Trainning phase (lr = {perceptron.lr})\nEpoch : {epoch}\nNº data : {t+1}")
    ax.plot(points1[:,0], points1[:,1], "b.")
    ax.plot(points2[:,0], points2[:,1], "r.")
    ax.plot([-6,6], [p(-6),p(6)], "k-")  # recta del perceptron
    ax.plot([-6,6], [recta(-6),recta(6)], "k-", alpha=0.4)  # recta objetivo
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    
    x, y = X[t], Y[t]                  
    perceptron.fit(x, y)

def train(X, Y, perceptron, epochs, interval=250, figsize=(6,6), save_filename="ani.gif"):

    n = len(Y)
    points1 = X[[f(p) > 0 for p in X]]
    points2 = X[[f(p) <= 0 for p in X]]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ani = FuncAnimation(fig, update, frames=range(epochs*n), fargs=[X, Y, perceptron, ax, points1, points2], interval=interval)
    ani.save(f"images/{save_filename}")
    plt.show()


def train_random_data():

    # ask for parameters
    print("Number of points:")
    n = 50 #ask_number()
    print("Number of epochs:")
    epochs = 3 #ask_number()
    print("Learning rate:")
    lr = 0.1 #ask_number(float)

    # points 
    np.random.seed(0)
    px = 10*np.random.rand(n) - 5
    py = 10*np.random.rand(n) - 5

    points = list(zip(px, py))

    # training data
    X = np.array(points)
    Y = np.array([classify(p) for p in points])

    perceptron = Perceptron(n_inputs=2, lr=lr)

    train(X, Y, perceptron, epochs=epochs, interval=100, save_filename="train_random_data.gif")


def train_3_points():
    E1, E2, E3 = (1,1), (1,0), (0,1)
    perceptron = Perceptron(n_inputs=2, lr=0.5)
    X = np.array([E1, E2, E3])
    Y = np.array([classify(p) for p in X])
    train(X, Y, perceptron, epochs=5, interval=1000, save_filename="train_3_points.gif")


if __name__ == "__main__":
    # train_3_points()
    train_random_data()
