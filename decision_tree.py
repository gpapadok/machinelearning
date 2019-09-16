import pprint

from sklearn import datasets
import numpy as np


def calculate_entropy(y):
	n_samples = y.shape[0]
	_, counts = np.unique(y, return_counts=True)
	entropy = 0
	for c in counts:
		entropy -= c / n_samples * np.log2(c / n_samples)
	return entropy


def decision_tree_fit(data, target, max_depth=4, feature_names=None, target_names=None):
	tree = dict()
	classes = np.unique(target)
	if classes.shape[0] <= 1:
		tree['class'] = classes[0]
		try:
			tree['name'] = target_names[tree['class']]
		except TypeError:
			pass
		return tree
		
	entropy_all = calculate_entropy(target)

	round_vec = np.vectorize(round)
	
	max_info_gain = 0
	max_feature = None
	split_point = None
	for idx in range(data.shape[1]):
		f = data[:,idx]
		try:
			for split in round_vec(np.arange(f.max(), f.min(), -.1), 1):
				# split dataset
				idx_left = f < split
				idx_right = idx_left ^ True
				data_left = data[idx_left]
				data_right = data[idx_right]
				target_left = target[idx_left]
				target_right = target[idx_right]
				if target_left.size == 0 or target_right.size == 0:
					continue

				# calculate entropy and information gain
				entropy1 = calculate_entropy(target_left)
				entropy2 = calculate_entropy(target_right)
				p_left = data_left.shape[0] / float(f.shape[0])
				p_right = data_right.shape[0] / float(f.shape[0])
				entropy_f = p_left * entropy1 + p_right * entropy2
				info_gain = entropy_all - entropy_f

				# compare to max information gain and replace if bigger
				if info_gain > max_info_gain:
					max_info_gain = info_gain
					max_feature = idx
					split_point = split
		except ValueError:
			pass
	# print (f'split at feature: {max_feature} and point {split_point}')
	if max_info_gain <= .1 or max_depth == 0:
		u, c = np.unique(target, return_counts=True)
		tree['class'] = u[c.argmax()]
		try:
			tree['name'] = target_names[tree['class']]
		except TypeError:
			pass
		return tree
	
	tree['split_point'] = split_point
	tree['feature'] = max_feature
	try:
		tree['name'] = feature_names[tree['feature']]
	except TypeError:
		pass
	tree['information_gain'] = max_info_gain
	idx_left_slice = data[:,max_feature] < split_point
	idx_right_slice = idx_left_slice ^ True
	tree['left'] = decision_tree_fit(data[idx_left_slice], target[idx_left_slice], max_depth-1, feature_names, target_names)
	tree['right'] = decision_tree_fit(data[idx_right_slice], target[idx_right_slice], max_depth-1, feature_names, target_names)
	return tree

	
def decision_tree_classify(sample, tree):
	try:
		return tree['class']
	except KeyError:  
		feature = tree['feature']
		split_point = tree['split_point']
		if sample[feature] < split_point:       
			return decision_tree_classify(sample, tree['left'])
		else:
			return decision_tree_classify(sample, tree['right'])


		
def main():
	iris = datasets.load_iris()

	X = iris.data[:,2:]
	y = iris.target

	classification_tree = decision_tree_fit(X, y, max_depth=5, feature_names=iris.feature_names[2:], target_names=iris.target_names)

	pprint.pprint(classification_tree)

	prediction = [decision_tree_classify(sample, classification_tree) for sample in X]

	print()
	print(f'Samples classified correctly: {sum(prediction==y)}')
	print(f'Accuracy: {sum(prediction==y)/X.shape[0] * 100}%')


if __name__=='__main__':
	main()