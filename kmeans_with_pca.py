from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def pca(X, n):
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    U, S, V = np.linalg.svd(np.cov(X.T))
    Ureduce = U[:,:n]
    Z = X @ Ureduce
    return Z

def eucli_dist(a, b):
    return np.sum(np.square(a-b))

def kmeans(X, n_clusters):
    n_samples, n_features = X.shape

    # initialiaze centroids
    rand_indices = np.random.choice(range(n_samples), size=n_clusters, replace=False)
    centroids = X[rand_indices,:]

    
    y = np.zeros((n_samples,1)).reshape(-1)


    for iter in range(1000):
        # assign samples to clusters
        for j in range(n_samples):
            sample = X[j,:]
            # calculate distance from each centroid
            dist = []
            for c in centroids:
                dist.append( eucli_dist(sample, c) )

            y[j] = np.argmin(dist)

        # find new centroids
        for j in range(n_clusters):
            A = X[y==j,:]
            centroids[j] = np.sum(A, axis=0) / A.shape[0]
            # centroids[j] = np.sum(X[y==j,:], axis=0) / X[y==j,:].shape[0]

    return y


def plot3d(X, y, iris):
    A = X[y==0,:]
    B = X[y==1,:]
    C = X[y==2,:]

    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')

    ax1.scatter(A[:,0], A[:,1], A[:,2])
    ax1.scatter(B[:,0], B[:,1], B[:,2])
    ax1.scatter(C[:,0], C[:,1], C[:,2])

    ax1.set_title('ground truth')
    ax1.legend((iris.target_names))

    y_clusters = kmeans(X, 3)

    A = X[y_clusters==0,:]
    B = X[y_clusters==1,:]
    C = X[y_clusters==2,:]

    ax2.scatter(A[:,0], A[:,1], A[:,2])
    ax2.scatter(B[:,0], B[:,1], B[:,2])
    ax2.scatter(C[:,0], C[:,1], C[:,2])

    ax2.set_title('kmeans')


    plt.show()





def main():
    iris = datasets.load_iris()

    X = pca(iris.data, 3)
    y = iris.target

    
    plot3d(X, y, iris)


if __name__=='__main__':
    main()