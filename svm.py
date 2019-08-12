import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets


def svm(X, y, lr=0.01):
	n_samples, n_features = X.shape
	X = np.c_[np.ones((n_samples,1)), X]
	w = np.random.rand(n_features+1) 
	for iteration in range(5000):		
		for sample, cat in zip(X, y):
			w -= 2 * lr * w * ( 1 / (iteration+1) )
			if cat == 1 and np.dot(w, sample) <= 1:
				w += lr * sample
			elif cat == 0 and np.dot(w, sample) >= -1:
				w -= lr * sample

	return w


def main():
	iris = datasets.load_iris()

	X = iris.data[:100,:2]
	y = iris.target[:100]

	w = svm(X,y,0.01)

	plt.scatter(X[y==0,0], X[y==0,1], marker='o', color='blue')
	plt.scatter(X[y==1,0], X[y==1,1], marker='o', color='green')
	plt.legend((iris.target_names[0], iris.target_names[1]))
	plt.xlabel(iris.feature_names[0])
	plt.ylabel(iris.feature_names[1])

	u = np.linspace(4.4,7)
	v = (-w[1] * u - w[0]) / w[2]
	plt.plot(u,v,'k')
	plt.show()

if __name__=='__main__':
	main()
