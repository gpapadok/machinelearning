import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

def calculate_distance(a, b):
	return np.sum((a-b)**2)


def k_nearest_neighbors_classify(sample, X, y, k=3):
	# calculate distances
	distances = [calculate_distance(sample, x) for x in X]

	# sort distances
	distances, y = zip(*sorted(zip(distances, y)))
	
	k_nearest_neighbors = y[1:k+1]
	u, c = np.unique(k_nearest_neighbors, return_counts=True)	

	return u[np.argmax(c)]


def main():
	iris = datasets.load_iris()

	n_samples = iris.target.shape[0]
	k_range = range(1, 10)

	# train with sepal width and length
	accuracy = [sum([k_nearest_neighbors_classify(sample, iris.data[:,:2], iris.target, k) for sample in iris.data[:,:2]]==iris.target) / float(n_samples) for k in k_range]

	fig = plt.figure()
	fig.suptitle('Accuracy at different values of k')

	ax2 = fig.add_subplot(311)
	ax2.plot(k_range, accuracy)
	plt.title('sepals')
	ax2.set_xticklabels([])

	# train with petal width and length
	accuracy = [sum([k_nearest_neighbors_classify(sample, iris.data[:,2:], iris.target, k) for sample in iris.data[:,2:]]==iris.target) / float(n_samples) for k in k_range]

	ax2 = fig.add_subplot(312)
	ax2.plot(k_range, accuracy)
	plt.ylabel('accuracy score')
	plt.title('petals')
	ax2.set_xticklabels([])

	# train with all features
	accuracy = [sum([k_nearest_neighbors_classify(sample, iris.data, iris.target, k) for sample in iris.data]==iris.target) / float(n_samples) for k in k_range]

	ax3 = fig.add_subplot(313)
	ax3.plot(k_range, accuracy)
	plt.xlabel('k value')
	plt.title('all features')

	fig.tight_layout()
	fig.subplots_adjust(top=.88)
	
	plt.show()
	
	

if __name__=='__main__':
	main()