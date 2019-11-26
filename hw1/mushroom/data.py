import pandas as pd
import pandas
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

def show_table(data):
    data.plot.bar(rot=0, subplots=True)
    plt.show()

if __name__ == "__main__":
    data = read_data("agaricus-lepiota.data")

    tables = all_feature_freq(data)

    for table in tables:
        print(tables[table])
        print()

    # test = []

    # for col in data.columns:
    #     test.append(data[col][512])
    
    # ans = naiveBayes.naive_bayes_classifier(tables, test)
    # print(ans)
