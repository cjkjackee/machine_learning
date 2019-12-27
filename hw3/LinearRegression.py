import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def inverse(data: pd.DataFrame):
    ans = pd.DataFrame(index=data.index, columns=data.columns)
    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if row == col:
                ans[col][row] = 1
            else:
                ans[col][row] = 0
    name = ['x'+str(i) for i in data.index]
    ans.columns = name
    tmp = gauss_jordanElimination(data.join(ans))
    tmp = tmp.drop(columns=data.columns)
    tmp.columns = data.columns
    return tmp

def gauss_jordanElimination(data: pd.DataFrame):
    data = data.transpose()
    for i in data.columns:
        data[i] /= data[i][i]
        for other in data.columns:
            if other is i:
                continue
            data[other] -= data[i]*data[other][i]
    data = data.transpose()
    return data

class LinearRegression:
    data = None
    error = None
    line = None

    def __init__(self, data=None, error=None, line=None):
        self.data = data
        self.error = error
        self.line = line
    
    def fitLine(self, n):
        a = pd.DataFrame()
        b = pd.DataFrame()
        for i in reversed(range(0,n)):
            a[i] = self.data[self.data.columns[0]]**i
        b['y'] = self.data[self.data.columns[1]]

        a_t = a.transpose()
        a_p = a_t.dot(a)
        a_p = inverse(a_p)
        a_p = a_p.dot(a_t)
        self.line = pd.Series(a_p.dot(b)['y'], name='line')

    def printLine(self):
        print('Fitting line:', end=' ')
        for i in reversed(range(0,self.line.shape[0])):
            if i is 0:
                print(self.line[i])
            else:
                print('{}x^{}'.format(self.line[i], i), end=' + ')
    
    def totalError(self):
        ans = pd.DataFrame()
        ans['predict'] = [0]*self.data.shape[0]
        for i in self.line.index:
            ans['predict'] += self.line[i]*(self.data[self.data.columns[0]]**i)
        ans['actual'] = self.data[self.data.columns[1]]
        ans['error'] = (ans['actual'] - ans['predict'])**2
        return ans['error'].sum()

    def plot(self, line=False):
        self.data.plot.scatter(x=self.data.columns[0], y=self.data.columns[1], c='red')
        if line is True:
            tmp_max = self.data.max()
            tmp_min = self.data.min()
            x = np.linspace(tmp_min[self.data.columns[0]], tmp_max[self.data.columns[0]], 1000)
            y = 0
            for i in self.line.index:
                y += self.line[i]*(x**i)
        plt.plot(x,y)
        plt.show()
