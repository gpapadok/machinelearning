from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, Theta1, Theta2):
    m = X.shape[0]
    a1 = X
    a2 = sigmoid(np.matmul(Theta1, a1.T)).T
    a2 = np.append(np.ones( (len(a2),1) ), a2, axis=1)
    a3 = sigmoid(np.matmul(Theta2, a2.T))
    return a3.T

def backpropagation(X, y, Theta1, Theta2, lr, debug=False):
    m = X.shape[0]

    # FORWARD PROP
    a1 = X
    a2 = sigmoid(np.matmul(Theta1, a1.T)).T
    a2 = np.append(np.ones( (len(a2),1) ), a2, axis=1)
    a3 = sigmoid(np.matmul(Theta2, a2.T)).T

    # BACKPROP

    d3 = y - a3

    if debug==True:
        print (np.sum(np.square(d3)))

    d2 = a2[:,1:] * (1 - a2[:,1:]) * np.matmul(d3, Theta2[:,1:])

    D1 = np.matmul(d2.T, a1)
    D2 = np.matmul(d3.T, a2)

    Theta1 = Theta1 + lr * D1
    Theta2 = Theta2 + lr * D2

    return Theta1, Theta2

def train(X, y, Theta1, Theta2, lr=0.001, iter=10000, debug=False):
    for j in range(iter):
        Theta1, Theta2 = backpropagation(X, y, Theta1, Theta2, lr, debug=debug)
    return Theta1, Theta2


def main():
    # Fetch data
    iris = datasets.load_iris()
    numofclass = max(iris.target) + 1
    numofexamples = len(iris.data)
    numoffeatures = iris.data.shape[1]

    X = np.append(np.ones( (numofexamples,1) ), iris.data, axis=1)

    y = np.zeros( (numofexamples,numofclass) )
    y[np.argwhere(iris.target==0), 0] = 1
    y[np.argwhere(iris.target==1), 1] = 1
    y[np.argwhere(iris.target==2), 2] = 1

    # Construct neural net

    input_nodes = numoffeatures
    hidden_nodes = input_nodes + 10
    output_nodes = numofclass

    Theta1 = np.random.random([hidden_nodes, input_nodes+1]) - .5
    Theta2 = np.random.random([output_nodes, hidden_nodes+1]) - .5

    # Train
    Theta1, Theta2 = train(X, y, Theta1, Theta2, lr=0.01, iter=10000, debug=False)

    # Calculate accuracy
    pred = forward_propagation(X, Theta1, Theta2)
    
    s1 = sum(pred[:50,0] > 0)
    s1 = [np.all(np.round(foo) == bar) for foo, bar in zip(pred[:50,:], y[:50])]
    s1 = sum(s1)
    print(f'Number of {iris.target_names[0]} samples classified correctly: {s1}')
    s2 = sum(pred[51:100,1] > 0)
    s2 = [np.all(np.round(foo) == bar) for foo, bar in zip(pred[50:100,:], y[50:100])]
    s2 = sum(s2)
    print(f'Number of {iris.target_names[1]} samples classified correctly: {s2}')
    s3 = sum(pred[101:,2] > 0)    
    s3 = [np.all(np.round(foo) == bar) for foo, bar in zip(pred[100:,:], y[100:])]
    s3 = sum(s3)
    print(f'Number of {iris.target_names[2]} samples classified correctly: {s3}')

    print(f'Total number of samples classified correctly: {s1+s2+s3}')
    print(f'Accuracy: {round((s1+s2+s3)/1.5, 2)}%')




if __name__=='__main__':
    main()