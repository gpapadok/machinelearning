import numpy as np
import matplotlib.pyplot as plt


def costfunction(X, y, theta):
    """ calculates cost function at specific theta """
    
    h = np.matmul(X, theta)
    return np.matmul((h-y).T, h-y) / 2 / np.shape(X)[0]

def linreg(X, y, theta=None, alpha=0.01, debug=False, iter=10000):
    """ linear regression iterative method """
    m = len(X)    

    cost = []

    def gradientdescent(theta):
        """ calculates theta gradient and performs gradient descent """
        h = np.matmul(X, theta)
        grad = np.matmul(X.T, h - y) / m
        theta = theta - alpha * grad
        cost.append(costfunction(X, y, theta))
        if debug == True:
            # print(cost[-1], grad[0], grad[1])
            print (cost[-1])
        return theta
        
    try:
        for j in range(iter):
            theta = gradientdescent(theta)
    except:
        import traceback
        traceback.print_exc()
    finally:
        return theta, cost

def normalequation(X, y):
    """ Calculates theta at local optimum with normal equation.
        Preferable when we have less than 10000 training examples. """
    m = len(X)
    try:
        # if X.T * X matrix is singular raises exception
        return np.matmul( (np.matmul( np.linalg.inv(np.matmul(X.T, X)), X.T )), y)
    except LinAlgError as err:
        print(err)


def intocm(inches):
    return inches * 2.54


def readData():
    """ Opens txt file, reads height data, and returns two numpy arrays """
    with open('Pearson.txt', 'r') as file:
        heights = file.read()
        heights = heights.split("\n")
        father = []
        son = []
        for data in heights[1:]:
            temp = data.split()
            if temp:
                # if float(temp[0]) < 10. or float(temp[1]) < 10.: continue
                father.append(float(temp[0]))
                son.append(float(temp[1]))

        return np.array(father), np.array(son)


def main():
    father, son = readData()
    theta = np.random.rand(2)

    # add bias terms
    father = np.append(np.ones((len(father),1)), father.reshape(-1,1), axis=1)

    # weights calculated with linear regression
    theta1, _ = linreg(father, son, theta, 0.0003, debug=False, iter=100000)
    print('theta from linreg = ')
    print(theta1)

    # weights calculated with normal equation
    theta2 = normalequation(father, son)
    print('theta from normalequation = ')
    print(theta2.reshape(-1))

    # plot data
    plt.plot(father[:,1], son, 'bo', markersize=2)

    # plot data and fitted line
    x = np.arange(np.min(father[:,1]), np.max(father[:,1])).reshape(-1,1)
    x = np.append( np.ones( (len(x),1) ), x, axis=1 )

    y2 = np.matmul( x, theta2 )
    y1 = np.matmul( x, theta1 )

    plt.plot(x[:,1], y1, 'k')
    plt.xlabel('father height(in)')
    plt.ylabel('son height(in)')
    plt.title('father / son heights')
    plt.show()



if __name__=='__main__':
    main()
