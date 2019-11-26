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

def split_data(data, size):
    msk = np.random.rand(len(data)) < size
    return data[msk], data[~msk]

def Kfold(data, k):
    li = []
    size = int(data.shape[0]/k)

    for i in range(0, k):
        tmp = data.sample(n=size)
        li.append(tmp)
        data = data.drop(tmp.index)
    
    while data.shape[0] is not 0:
        for i in range(0, k):
            tmp = data.sample(n=1)
            li[i] = li[i].append(tmp)
            data = data.drop(tmp.index)
            if int(data.shape[0]) is 0:
                break
    return li

def gen_freq(table, sum):
    for col in table.columns:
        for index in table.index:
            table[col][index] /= sum[col]
    table = table.replace(float(0),  np.nan)
    return table

def all_feature_freq(data):
    tmp = data.groupby([0]).size().astype('float64').fillna(0)
    tmp1 = data.groupby([0]).size().astype('float64').fillna(0)
    total = tmp.sum()
    for col in tmp.index:
        tmp[col] /= total

    table = {0:tmp}

    for col in data.columns:
        if col is 0:
            continue
        tmp = gen_table(data, col)
        tmp = gen_freq(tmp, tmp1)
        if tmp is None:
            continue
        table.update({col: tmp})
    return table

def predict(freq_tables, test):
    li = []
    ans = ['e', 'p']
    predictE = math.log10(freq_tables[0].e)
    predictP = math.log10(freq_tables[0].p)

    for index in test.index:
        probE = predictE
        probP = predictP

        for col in test.columns:
            if col is 0:
                continue

            table = freq_tables[col]
            feature = test[col][index]

            if not np.isnan(table['e'][feature]):
                # print("{},e,{}: {}".format(col, feature, table['e'][feature]))
                probE += math.log10(table['e'][feature])
            if not np.isnan(table['p'][feature]):
                # print("{},p,{}: {}".format(col, feature, table['p'][feature]))
                probP += math.log10(table['p'][feature])
        li.append(ans[int(np.argmax([probE, probP]))])
    return li

def gen_confusion_matrix(actual, predict):
	actual_case = pd.Series(actual, name='actual')
	predict_case = pd.Series(predict, name="predict")
	return pd.crosstab(predict_case, actual_case)

def performances(confusion_matrix):
    tp = confusion_matrix['e']['e']
    tn = confusion_matrix['p']['p']
    fp = confusion_matrix['p']['e']
    fn = confusion_matrix['e']['p']
    return {'accuracy': (tp+tn)/(tp+tn+fp+fn), 'sensitivity': (tp)/(tp+fn), 'precision': (tp)/(tp+fp)}

def show_table(data):
    data.plot.bar(rot=0, subplots=True)
    plt.show()


if __name__ == "__main__":
    data = read_data('agaricus-lepiota.data')
    data = shuffle_data(data)

    # hold out
    train,test = split_data(data, 0.7)

    actual = test[0].tolist()
    test = test.drop(columns=[0])
    
    freq = all_feature_freq(train)

    matrix = gen_confusion_matrix(actual, predict(freq, test))
    print()
    print()
    print("Holdout validation with the ratio 7:3: ")
    print(matrix)
    print(performances(matrix))
    print()

    # print(freq)

    # print(predict(freq, test))

    # K fold
    print('K-fold cross-validation with k=3:')
    data_set = Kfold(data, 3)
    tmp = {'accuracy': 0, 'sensitivity': 0, 'precision': 0}
    for i in range(len(data_set)):
        train = list(data_set)
        del train[i]
        train = pd.concat(train)
        
        test = data_set[i]
        actual = test[0].tolist()
        test = test.drop(columns=[0])

        matrix = gen_confusion_matrix(actual, predict(all_feature_freq(train), test))

        print('validation data set {}:'.format(i))
        print(matrix)
        print()
        for k, v in performances(matrix).items():
            tmp[k] += v
    print({k : v/3 for k,v in tmp.items()})