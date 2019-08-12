from sklearn import datasets
import numpy as np


def predict(X, weight):
    return np.matmul(weight, X.T).T

def train(X, y, weight, lr, iter=5000):
    m = X.shape[0]

    for j in range(iter):
        out = predict(X, weight)

        # Update weights for each sample
        for j in range(m):
            cl = np.argwhere(y[j] == 1)[0][0]
            index1 = cl
            index2 = (cl + 1) % 3
            index3 = (cl + 2) % 3
            if out[j,index1] <= 0:
                weight[index1,:] = weight[index1,:] + lr * X[j,:]
            if out[j,index2] > 0:
                weight[index2,:] = weight[index2,:] - lr * X[j,:]
            if out[j,index3] > 0:
                weight[index3,:] = weight[index3,:] - lr * X[j,:]

            # print (np.sum(np.square(y-out)))

    return weight


def main():
    # Initialize data
    iris = datasets.load_iris()
    n_classes = max(iris.target) + 1
    n_samples, n_features = iris.data.shape

    X = np.append(np.ones( (n_samples,1) ), iris.data[:,:n_features], axis=1)

    y = np.zeros( (n_samples,n_classes) )            
    y[np.argwhere(iris.target==0), 0] = 1
    y[np.argwhere(iris.target==1), 1] = 1
    y[np.argwhere(iris.target==2), 2] = 1

    weights = np.random.random([n_classes, n_features+1]) - .5 

    # Train
    weights = train(X, y, weights, 0.01)

    # Calculate accuracy
    pred = predict(X, weights)    
    
    s1 = sum(pred[:50,0] > 0)
    print(f'Number of {iris.target_names[0]} samples classified correctly: {s1}')
    s2 = sum(pred[51:100,1] > 0)
    print(f'Number of {iris.target_names[1]} samples classified correctly: {s2}')
    s3 = sum(pred[101:,2] > 0)
    print(f'Number of {iris.target_names[2]} samples classified correctly: {s3}')

    print(f'Total number of samples classified correctly: {s1+s2+s3}')
    print(f'Accuracy: {round((s1+s2+s3)/1.5, 2)}%')



if __name__=='__main__':
    main()