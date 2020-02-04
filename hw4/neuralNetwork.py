import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Layer:
    neuron = None
    cache = None

    def __init__(self, neuron=None, input_col=None):
        if isinstance(neuron, int):
            weight = pd.DataFrame()
            col = [i for i in range(0,input_col)]
            col.append('c')
            for i in range(0,neuron):
                weight = weight.append(pd.Series([random.uniform(-1.0,1.0) for i in range(0,input_col+1)], index=col), ignore_index=True)
            self.neuron = weight.transpose()
        elif isinstance(neuron, list):
            self.neuron = neuron
        else:
            pass
    
    def clearCache(self):
        del self.cache
    
    def output(self, data, caching=False):
        ans = data.dot(self.neuron)
        if caching:
            self.cache = ans
        return ans

class Network:
    layers = None

    def __init__(self, layers=None, input_number=None, output_number=None, layer_number=None, neuron_number=None):
        if isinstance(layers, list):
            self.layers = layers
        else:
            # first hidden layer has input_number input
            tmp = [Layer(neuron=neuron_number, input_col=input_number)]
            # other hidden layer has neuron_number input
            for i in range(0,layer_number-1):
                tmp.append(Layer(neuron=neuron_number, input_col=neuron_number))
            tmp.append(Layer(neuron=output_number, input_col=neuron_number))
            self.layers = tmp

    def backPropagation(self, loss):
        for i in reversed(range(0,len(self.layers))):
            data = self.layers[i].cache
            grad = loss_dw(loss, data, sigmoid(self.layers[i-1].cache))
            grad_c = loss_db(loss, data)
            grad_c.name = 'c'
            grad = grad.append(grad_c)
            self.layers[i].neuron -= grad
            loss = loss_da(loss, data, self.layers[i].neuron)

    def forwardPropagation(self, data, caching=False):
        tmp = pd.DataFrame()
        tmp['c'] = [1]*data.shape[0]
        for layer in self.layers:
            data = layer.output(data.join(tmp), caching=caching)
        return sigmoid(data)

    def loss(self, p, target):
        column = target.groupby(target).size().index.to_list()
        p['target'] = target
        for col in column:
            tmp = list(column)
            tmp.remove(col)
            a = [0]*len(tmp)
            tmp.append(col)
            a.append(1)
            p['target_'+str(col)] = p['target'].replace(tmp, a)

        ans = pd.DataFrame()
        for col, label in zip(self.layers[-1].neuron.columns, column):
            ans[col] = (p[col] - p['target_'+str(label)])
        return ans

    def showLayers(self):
        for n in self.layers:
            print(n.neuron)
    
    def train(self, data, epochs=500000, show=1000):
        target = data['target']
        data = data.drop(columns=['target'])
        data = data.transpose()
        data = data.reset_index(drop=True)
        data = data.transpose()

        for e in range(0, epochs):
            self.forwardPropagation(pd.DataFrame(data), caching=True)
            loss = self.loss(sigmoid(self.layers[-1].cache), target)
            self.backPropagation(loss)
            if not(e % show):
                print('epochs: {}, loss: {}'.format(e, abs(loss).sum().sum()/loss.shape[0]))

        for layer in self.layers:
            layer.clearCache()

    def predict(self,data):
        data = data.transpose()
        data = data.reset_index(drop=True)
        data = data.transpose()

        ans = []
        data = self.forwardPropagation(data)
        for row in data.index:
            ans.append(data.iloc[row].idxmax())
        return ans
    
    def save(self, f):
        with open(f,'wb') as fd:
            pickle.dump(self.layers, fd)


    
def load(f):
    with open(f, 'rb') as fd:
        li = pickle.load(fd)
    return Network(layers=li)

def genNetwork(input_number: int, output_number:int, layer_number=2, neuron_number=5):
    return Network(input_number=input_number-1, output_number=output_number, layer_number=layer_number, neuron_number=neuron_number)

def sigmoid(n):
    return 1.0/(1.0 + np.exp(-n))

def sigmoid_deriv(n):
    return sigmoid(n)*(1.0-sigmoid(n))

def loss_dw(s_ans, ans, s_a):
    return s_a.transpose().dot((2*s_ans)*sigmoid_deriv(ans))
def loss_da(s_ans, ans, w):
    w = w.drop(index=['c'])
    return ((2*s_ans)*sigmoid_deriv(ans)).dot(w.transpose())
def loss_db(s_ans, ans):
    return ((2*s_ans)*sigmoid_deriv(ans)).sum()

def accuracy(predict, actual):
    cm = pd.crosstab(pd.Series(predict, name='predict'), pd.Series(actual, name='actual'))
    acc = 0
    for x in cm.index:
        acc += cm[x][x]
    acc /= cm.sum().sum()
    print(cm)
    print('accuracy: {}'.format(acc))

def plot(data1=None, data2=None):
    ax = data1.plot.scatter(x=data1.columns[0], y=data1.columns[1], c='red')
    data2.plot.scatter(x=data2.columns[0], y=data2.columns[1], c='blue', ax=ax)
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('data.txt', header=None, names=['x', 'y', 'target'])

    # pos = data[data[2] == 1]
    # neg = data[data[2] == 0]
    # plot(pos, neg)

    actual = data['target']

    n = genNetwork(data.shape[1], 2)
    n.showLayers()
