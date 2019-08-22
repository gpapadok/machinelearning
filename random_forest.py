import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

from decision_tree import decision_tree_fit, decision_tree_classify

def bagging(X, y):
	n_samples = X.shape[0]

	idx = np.random.randint(0, n_samples, size=(n_samples,))

	return X[idx], y[idx]


def random_forest_fit(X, y, n_trees=100):
	forest = []
	for _ in range(n_trees):
		features = np.random.choice(4, np.random.randint(2, 4), replace=False)
		tree = decision_tree_fit(*bagging(X[:,features], y))
		tree['features_used'] = features
		forest.append(tree)
	
	return forest

def random_forest_predict(sample, forest):
	ensemble = [decision_tree_classify(sample[tree['features_used']], tree) for tree in forest]
	u, c = np.unique(ensemble, return_counts=True)
	return u[np.argmax(c)]


def main():
	iris = datasets.load_iris()

	X = iris.data
	y = iris.target

	n_samples = X.shape[0]
	n_trees = 100

	forest = random_forest_fit(X, y, n_trees)
	predictions = [random_forest_predict(sample, forest) for sample in X]
	# print(f'Samples classified correctly: {sum(predictions==y)}')

	n_trees_range = [j for j in range(1, n_trees)]
	accuracy = []
	for k in n_trees_range:
		predictions = [random_forest_predict(sample, forest[:k]) for sample in X]
		accuracy.append(sum(predictions==y) / n_samples)


	plt.plot(range(1,n_trees), accuracy)
	plt.xlabel('k trees')
	plt.ylabel('accuracy')
	plt.xlim([1, n_trees])
	plt.title('Accuracy score for different number of trees')
	plt.show()


if __name__=='__main__':
	main()