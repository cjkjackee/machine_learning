import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file):
    return pd.read_csv(file, sep=',', header=None).replace('?', np.nan).dropna(axis='columns')

def gen_table(data, feature):
    return data.groupby([feature, 0]).size().unstack().astype('float64').fillna(0)

def shuffle_data(data):
    return data.sample(frac=1)

def Kfold(data, k):
	li = []
	for i in range(0, k):
		tmp = data.sample(n=1)
		li.append(tmp)
		data = data.drop(tmp.index)
	
	while int(data.shape[0]) is not 0:
		for i in range(0, k):
			tmp = data.sample(n=1)
			li[i] = li[i].append(tmp)
			data = data.drop(tmp.index)
			if int(data.shape[0]) is 0:
				break
	return li

def split_data(data, size):
    msk = np.random.rand(len(data)) < size
    return data[msk], data[~msk]

def separate_by_name(data, col):
	table = {}
	tmp = data.groupby([col]).size()
	for name in tmp.index:
		table.update({name: data[data[col] == name].drop(columns=[4])})
	return table

def probability_of_types(data):
	table = {}
	total = 0
	for key in data.keys():
		table.update({key: data[key].shape[0]})
		total += data[key].shape[0]
	return {k: v/total for k,v in table.items()}

def average(data):
	table = {}
	for key in data.keys():
		table.update({key: data[key].mean()})
	return table

def standard_deviation(data):
	table = {}
	for key in data.keys():
		table.update({key: data[key].std()})
	return table

def normal_distribution(feature, avg, std):
	const = 1 / (std * math.sqrt(2*math.pi))
	power_num = -math.pow((feature-avg), 2) / (2*math.pow(std,2))
	return const*math.exp(power_num)

def predict(data, prob, avg, std):
	li = []
	keys = [x for x in prob.keys()]
	for index in data.index:
		ans = {k:math.log2(p) for k,p in prob.items()}
		for col in data.columns:
			for key in prob.keys():
				# print("data: {:<10} avg: {:<20} std: {:<20} nd: {}".format(data[col][index], avg[key][col], std[key][col], normal_distribution(data[col][index], avg[key][col], std[key][col])))
				try:
					ans[key] += math.log2(normal_distribution(data[col][index], avg[key][col], std[key][col]))
				except:
					pass
		li.append(keys[np.argmax([x for x in ans.values()])])
	return li

def gen_confusion_matrix(actual, predict):
	actual_case = pd.Series(actual, name='actual')
	predict_case = pd.Series(predict, name="predict")
	return pd.crosstab(predict_case, actual_case)

def model_accuracy(confusion_matrix, label):
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for col in confusion_matrix.columns:
		for index in confusion_matrix.index:
			if col is label and index is label:
				tp = confusion_matrix[col][index]
			elif col is label and index is not label:
				fn += confusion_matrix[col][index]
			elif index is label and col is not label:
				fp += confusion_matrix[col][index]
			else:
				tn += confusion_matrix[col][index]
	return {'accuracy': (tp+tn)/(tp+tn+fp+fn), 'sensitivity': (tp)/(tp+fn), 'precision': (tp)/(tp+fp)}

if __name__ == "__main__":
	table = read_data('iris.data')
	table = shuffle_data(table)

	# hold out 
	train, test = split_data(table, 0.7)

	train = separate_by_name(train, 4)

	test_ans = test[4]
	test = test.drop(columns=[4])

	result = predict(test, probability_of_types(train), average(train), standard_deviation(train))

	confusion_matrix = gen_confusion_matrix(test_ans.tolist(), result)

	print()
	print(confusion_matrix)
	for name in confusion_matrix.index:
		print("performance of {}: {}".format(name, model_accuracy(confusion_matrix, name)))
	print()

	# print("hold-out accuracy: {:.2%}".format())

	# for f, p in zip(test_ans, result):
	# 	b = "False"
	# 	if f == p:
	# 		b = "True"
	# 	print("data: {:20} predict: {:20}  {}".format(f, p, b))


	# k fold 
	k = 3
	data_set = Kfold(table, k)
	tmp = {'accuracy': float(0), 'sensitivity': float(0), 'precision': float(0)}
	perf = {x: tmp for x in table.groupby([4]).size().index}
	for i in range(len(data_set)):
		train = list(data_set)
		del train[i]
		train = pd.concat(train)
		train = separate_by_name(train, 4)

		test = data_set[i]
		test_ans = test[4]
		test = test.drop(columns=[4])

		result = predict(test, probability_of_types(train), average(train), standard_deviation(train))

		confusion_matrix = gen_confusion_matrix(test_ans.tolist(), result)

		print("validation data set {}:".format(i))
		print("confusion matrix:")
		print(confusion_matrix)
		print()

		for name in confusion_matrix.columns:
			perf[name] = {x: y+perf[name][x] for x,y in model_accuracy(confusion_matrix, name).items()}

	for name, item in perf.items():
		print("average performance of {}: {}".format(name, {x: v/k for x,v in item.items()}))
	print()
	