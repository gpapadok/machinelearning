from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import norm

def naive_bayes_fit(X, y):
	n_samples = y.shape[0]
	categories, counts = np.unique(y, return_counts=True)
	p_cat = counts / n_samples

	dist_list = []
	for f in X.T:
		means = [np.mean(f[y==cl]) for cl in categories]
		stds = [np.std(f[y==cl]) for cl in categories]
		dist_list.append(norm(loc=means, scale=stds))

	return dist_list, p_cat


def naive_bayes_predict(X, dist_list, p_cat):
	P = 1
	for dist, f in zip(dist_list, X.T):
		P *= dist.pdf(f)
	P *= p_cat
	return P.argmax()


def main():
	iris = datasets.load_iris()

	X = iris.data[:,:2]
	y = iris.target

	# TRAIN
	norms, p_cat = naive_bayes_fit(X, y)
	predictions = [ naive_bayes_predict(sample, norms, p_cat) for sample in X ]


	# CALCULATE ACCURACY
	correct_predictions = sum(y == predictions)
	accuracy = correct_predictions / X.shape[0]
	print (f'Number of samples classified correctly: {correct_predictions} out of {X.shape[0]}')
	print (f'Accuracy score: {accuracy*100} %')

	# PLOT
	plt.scatter(X[:50,0], X[:50,1])
	plt.scatter(X[50:100,0], X[50:100,1])
	plt.scatter(X[100:,0], X[100:,1])

	u = np.linspace(*plt.xlim())
	v = np.linspace(*plt.ylim())

	U, V = np.meshgrid(u, v)

	w = [ naive_bayes_predict(np.array([uu, vv]), norms, p_cat) for uu, vv in zip(np.ravel(U), np.ravel(V)) ]

	w = np.array(w)
	w = w.reshape(U.shape)


	plt.contourf(U, V, w, alpha=.3)
	plt.show()



if __name__=='__main__':
	main()