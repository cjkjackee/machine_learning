import pandas as pd
import math
from MLdata import Data

class node:
    is_numeric = False
    rule = None
    # variable for categorical node
    children = None
    unknow_child = None
    #  variable for continuous node
    mean = None

    def __init__(self, rule, is_numeric=False, mean=None, children=None, unknow=None):
        self.rule = rule
        self.is_numeric = is_numeric
        self.mean = mean
        self.unknow_child = unknow
        if children is not None:
            if isinstance(children, list):
                self.children = {k: None for k in children}
        else:
            if is_numeric:
                self.children = {'smaller': None, 'greater': None}

class randomForest:
    forest = []

    def __init__(self, forest=[]):
        self.forest = forest
    
    def gen(self, data: Data, size=3):
        self.forest.clear()
        for dataset in data.Kfold(size):
            self.forest.append(genTreeV2(dataset, 5))
        
        return self.forest
    
    def predict(self, data:Data):
        ans = []
        for tree in self.forest:
            ans.append(predict(tree, data))
        result = pd.DataFrame({i:v for i, v in zip(range(len(ans)), ans)}, index=data.index)
        ans.clear()
        for index in result.index:
            ans.append(result.loc[index].value_counts().idxmax())
        return ans




def genTree(data: Data):
    print()
    target_entropy = entropy(data.featureFrequency(col=data.target))

    if data.shape[1] is 1 and data.columns is data.target:
        return None

    li = []
    for feature in data.drop(columns=data.target):
        tmp = data.featureFrequency(col=feature)
        li.append(averageInformationEntropy(tmp))
    info_entropy = pd.Series(li, index=data.drop(columns=data.target).columns)
    info_gain = target_entropy - info_entropy
    rule = info_gain.idxmax()
    print(info_gain)
    print(rule)

    #TODO: all information gain is 0, return max count of target col as child
    #if info_gain.sum():
        #return data.groupby([data.target]).size().idxmax()

    #gen continuous data node
    if rule in data.numericalData(): 
        mean = data[rule].sum() / data.shape[0]
        now = node(rule, is_numeric=True, mean=mean)
        child_data = {'smaller': Data(data[data[rule] < mean], target=data.target), 'greater': Data(data[data[rule] >= mean], target=data.target)}
    # gen categorical data node
    else: 
        now = node(rule, children=data.featureFrequency(col=rule).index.to_list())
        child_data = {key: Data(data[data[rule]==key], target=data.target) for key in now.children.keys()}
    # gen children
    for key in child_data.keys():
        target_class = child_data[key].groupby([data.target]).size().index.to_list()
        print("{}: {}".format(key, target_class))
        print(child_data[key])
        print()
        if len(target_class) == 1:
            now.children[key] = target_class[0]
        else:
            now.children[key] = genTree(child_data[key].drop(columns=rule))

    return now

def genTreeV2(data: Data, level: int):
    # print()

    if level is 0:
        return data.groupby([data.target]).size().idxmax()

    target_entropy = entropy(data.featureFrequency(col=data.target))

    li = []
    for feature in data.drop(columns=data.target):
        tmp = data.featureFrequency(col=feature)
        li.append(averageInformationEntropy(tmp))
    info_entropy = pd.Series(li, index=data.drop(columns=data.target).columns)
    info_gain = target_entropy - info_entropy
    rule = info_gain.idxmax()
    # print(info_gain)
    # print(rule)

    #gen continuous data node
    if rule in data.numericalData(): 
        mean = data[rule].sum() / data.shape[0]
        now = node(rule, is_numeric=True, mean=mean)
        child_data = {'smaller': Data(data[data[rule] < mean], target=data.target), 'greater': Data(data[data[rule] >= mean], target=data.target)}
    # gen categorical data node
    else: 
        now = node(rule, children=data.featureFrequency(col=rule).index.to_list(), unknow=data.groupby([data.target]).size().idxmax())
        child_data = {key: Data(data[data[rule]==key], target=data.target) for key in now.children.keys()}
    # gen children
    for key in child_data.keys():
        target_class = child_data[key].groupby([data.target]).size().index.to_list()
        # print("{}: {}".format(key, target_class))
        # print(child_data[key])
        # print()
        if len(target_class) == 1:
            now.children[key] = target_class[0]
        else:
            now.children[key] = genTreeV2(child_data[key].drop(columns=rule),  level-1)

    return now

def predict(tree: node, data: Data):
    first = True
    li = []
    for row in data.index:
        now = tree
        test = data.loc[row]
        
        while isinstance(now, node):
            # if first:
            #     print(now.rule)
            if now.is_numeric:
                if test[now.rule] < now.mean:
                    now = now.children['smaller']
                else:
                    now = now.children['greater']
            else:
                if test[now.rule] not in now.children.keys():
                    now = now.unknow_child
                else: 
                    now = now.children[test[now.rule]]
        li.append(now)
        first = False
    return li

def modelAccuracy(confusion_matrix, label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for col in confusion_matrix.columns:
        for index in confusion_matrix.index:
            print('col: {}, index: {}, label: {}'.format(col, index, label))
            if (col is label) and (index is label):
                tp = confusion_matrix[col][index]
            elif str(col) is str(label) and str(index) is not str(label):
                fn += confusion_matrix[col][index]
            elif str(index) is str(label) and str(col) is not str(label):
                fp += confusion_matrix[col][index]
            else:
                tn += confusion_matrix[col][index]
    print("tp: {}, tn: {}, fp: {}, fn: {}".format(tp,tn,fp,fn))
    return {'accuracy': (tp+tn)/(tp+tn+fp+fn), 'sensitivity': (tp)/(tp+fn), 'precision': (tp)/(tp+fp)}

def entropy(row):
    ans = 0
    total = row.sum()
    for v in row.values:
        if int(v) is not 0:
            ans += -(v/total)*(math.log2(v/total))
    return ans

def averageInformationEntropy(data: Data):
    ans = 0
    total = data.sum().sum()

    for row in data.index:
        if data.loc[row].isnull().any():
            ans += 0
        else:
            ans += (data.loc[row].sum()/total)*entropy(data.loc[row])
    return ans

