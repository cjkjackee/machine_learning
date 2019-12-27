import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class LogisticRegression:
    data1 = None
    data2 = None
    weight = None
    error = None

    def __init__(self, data1=None,  data2=None, weight=None, error=None):
        if isinstance(weight, type(None)):
            self.weight = pd.Series([float(1)]*3, index=['x','y','c'])
        else:
            self.weight = weight

        self.data1 = data1
        self.data2 = data2
        self.error = error
    
    def sigmoid(self, w, x):
        return 1/(1+np.exp(-1*w.dot(x)))

    def L2Norm(self):
        li1 = [float(0)]*self.data1.shape[0]
        li2 = [float(1)]*self.data2.shape[0]
        data = self.data1.append(self.data2)
        data = data.reset_index(drop=True)
        data.columns = ['x', 'y']
        data['c'] = [1]*data.shape[0]
        data['predict'] = [self.sigmoid(self.weight, data.iloc[i]) for i in data.index]
        data['target'] = li1 + li2
        for i in data.drop(columns=['predict', 'target']).columns:
            data[i] *= (data['target']-data['predict'])*data['predict']*(1-data['predict'])
        return data.drop(columns=['predict', 'target'])

    def crossEntropy(self):
        li1 = [float(0)]*self.data1.shape[0]
        li2 = [float(1)]*self.data2.shape[0]
        data = self.data1.append(self.data2)
        data = data.reset_index(drop=True)
        data.columns = ['x', 'y']
        data['c'] = [1]*data.shape[0]
        data['predict'] = [self.sigmoid(self.weight, data.iloc[i]) for i in data.index]
        data['target'] = li1 + li2
        for i in data.drop(columns=['predict', 'target']).columns:
            data[i] *= (data['predict']-data['target'])
        return data.drop(columns=['predict', 'target'])

    def fitByL2Norm(self, data1=None, data2=None):
        if (isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame)):
            self.data1 = pd.DataFrame(data1)
            self.data2 = pd.DataFrame(data2)
        else:
            data1 = pd.DataFrame(self.data1)
            data2 = pd.DataFrame(self.data2)

        alpha = 0.002
        trig = 0

        while True:
            tmp_w = self.L2Norm()
            tmp_w = tmp_w.sum()
            tmp_error = self.totalError(self.weight +(alpha*tmp_w))
            print("error: {}, alpha: {}".format(tmp_error, alpha))
            if self.error is None or self.error > tmp_error:
                if self.error is not None:
                    rate = (self.error-tmp_error)*100
                    # print(rate)
                    if rate < 0.1:
                        alpha += alpha
                self.weight = self.weight +(alpha*tmp_w)
                self.error = tmp_error
            else:
                if trig > 10:
                    break
                else: 
                    alpha *= 0.01
                    trig += 1
    
    def fitbyCrossEntropy(self, data1=None, data2=None):
        if (isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame)):
            self.data1 = pd.DataFrame(data1)
            self.data2 = pd.DataFrame(data2)
        else:
            data1 = pd.DataFrame(self.data1)
            data2 = pd.DataFrame(self.data2)

        alpha = 0.2
        tmp_w = self.crossEntropy()
        tmp_w = tmp_w.sum() / tmp_w.shape[0]
        trig = 0
        while True:
            tmp_w = self.crossEntropy()
            tmp_w = tmp_w.sum() / tmp_w.shape[0]
            tmp_error = self.totalError(self.weight - (alpha*tmp_w))
            print("error: {}, alpha: {}".format(tmp_error, alpha))
            if self.error is None or self.error > tmp_error:
                if self.error is not None:
                    rate = (self.error-tmp_error)*100
                    # print(rate)
                    if rate < 0.005:
                        alpha += alpha
                self.weight = self.weight - (alpha*tmp_w)
                self.error = tmp_error
            else:
                break
                # if trig > 3:
                #     break
                # else:
                #     alpha /= 10
                #     trig += 1


    def totalError(self, weight):
        li1 = [float(0)]*self.data1.shape[0]
        li2 = [float(1)]*self.data2.shape[0]
        data = self.data1.append(self.data2)
        data.columns = ['x', 'y']
        data['c'] = [float(1)]*data.shape[0]
        data = data.reset_index(drop=True)
        data['predict'] = [self.sigmoid(weight, data.iloc[i]) for i in data.index]
        data = data.drop(columns=['c'])
        data['target'] = li1 + li2
        data['error'] = (data['target'] - data['predict'])
        data['error'] = data['error']**2
        return data['error'].sum() / 2

    def predict(self, data):
        data = data.reset_index(drop=True)
        data.columns = ['x', 'y']
        data['c'] = [float(1)]*data.shape[0]
        data['predict'] = [self.sigmoid(self.weight, data.iloc[i]) for i in data.index]
        return data[data['predict'] <= 0.5], data[data['predict'] > 0.5]

    def printWeight(self):
        print()
        for i in self.weight.index:
            if i is 'c':
                break
            print('{}{}'.format(self.weight[i], i), end=' + ')
        print(self.weight['c'])

    def plot(self, data1=None, data2=None):
        if not (isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame)):
            data1 = self.data1
            data2 = self.data2
        ax = data1.plot.scatter(x=data1.columns[0], y=data2.columns[1], c='red')
        data2.plot.scatter(x=data2.columns[0], y=data2.columns[1], c='blue', ax=ax)
        plt.show()
