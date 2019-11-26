import pandas as pd
import numpy as np

class Data(pd.DataFrame):
    numeric_col = None
    target = None

# overrided function
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, target=0, numeric_col=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self.numeric_col = numeric_col
        self.target = target

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        self._update_inplace(super().join(other, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort))
    
    def replace(self, to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad'):
        return Data(super().replace(to_replace=to_replace, value=value, inplace=inplace, limit=limit, regex=regex, method=method), numeric_col=self.numeric_col, target=self.target)
    
    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        return Data(super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors), numeric_col=self.numeric_col, target=self.target)
######

# data processing
    def setTarget(self, label):
        self.target = label

    def shuffle(self):
        self._update_inplace(self.sample(frac=1))
        return self

    def dropMissing(self, axis=0, miss_value=r'\?'):
        self._update_inplace(self.replace(regex=miss_value, value=pd.NaT).dropna(axis=axis))
        return self
    
    def fillMissing(self):
        self._update_inplace(self.replace(regex=r'\?', value=pd.NaT))

        if self.numeric_col is None:
            self.featureClassify()
        
        tmp = self.isnull().any()
        tmp = tmp[tmp == True] # get item which value is True from tmp

        for feature in tmp.index:
            if feature in self.numeric_col:
                tmp1 = self[feature].sum() / self.shape[0]
            else:
                tmp1 = self.groupby(feature).size()
                tmp1 = tmp1[tmp1 == max(tmp1.values)].index[0]

            self[feature] = self[feature].replace(pd.NaT, tmp1)

        return self
    
    def featureFrequency(self, col=None):
        if self.numeric_col is None:
            self.featureClassify()

        if col is None:
            all_col = []
            for col in self.columns:
                if col == self.target:
                    tmp = self.groupby([col]).size().div(self.shape[0])
                    all_col.append(tmp)
                elif col not in self.numeric_col:
                    tmp = self.groupby([col, self.target]).size().unstack()
                    total = tmp.sum(axis=1)
                    for col in tmp.columns:
                        tmp[col] = tmp[col].div(total)
                    all_col.append(tmp)
                else:
                    mean = self[col].sum() / self.shape[0]
                    tmp1 = self[self[col] < mean].groupby([self.target]).size()
                    tmp2 = self[self[col] >= mean].groupby([self.target]).size()
                    
                    li = []
                    for x in range(tmp1.shape[0]):
                        li.append('<{}'.format(int(mean)))
                    for x in range(tmp1.shape[0]):
                        li.append('>={}'.format(int(mean)))
                    li2 = tmp1.index.to_list()
                    li2.extend(tmp2.index.to_list())

                    s = tmp1.tolist()
                    s.extend(tmp2.tolist())

                    tmp = pd.DataFrame({col:li , self.target:li2, 'sum':s})
                    tmp = tmp.pivot(index=col, columns=self.target, values='sum')
                    all_col.append(tmp)
            return all_col
        else:
            if col == self.target:
                tmp = self.groupby([col]).size()
            elif col not in self.numeric_col:
                    tmp = self.groupby([col, self.target]).size().unstack()
            else:
                mean = self[col].sum() / self.shape[0]
                smaller = self[self[col] < mean].groupby([self.target]).size()
                greater = self[self[col] >= mean].groupby([self.target]).size()
                
                x = []
                for i in self.groupby([self.target]).size().index:
                    x.append(0)
                tmp1 = pd.Series(x, index=self.groupby([self.target]).size().index)
                tmp2 = pd.Series(x, index=self.groupby([self.target]).size().index)

                for i in smaller.index:
                    tmp1[i] = smaller[i]
                for i in greater.index:
                    tmp2[i] = greater[i]

                # print(tmp1)
                # print(tmp2)
                # print()

                li = []
                for x in range(tmp1.shape[0]):
                    li.append('<{}'.format(int(mean)))
                for x in range(tmp1.shape[0]):
                    li.append('>={}'.format(int(mean)))
                li2 = tmp1.index.to_list()
                li2.extend(tmp2.index.to_list())

                s = tmp1.tolist()
                s.extend(tmp2.tolist())

                # print("li: {}\nli2: {}\ns: {}".format(li, li2, s))

                tmp = pd.DataFrame({col:li , self.target:li2, 'sum':s})
                tmp = tmp.pivot(index=col, columns=self.target, values='sum')
            return tmp

    def featureClassify(self):
        # classify feature as numerical or categorical data
        if self.numeric_col is not None:
            return self.numeric_col

        self.numeric_col = []

        if self.empty:
            return []

        tmp = self.sum()
        for index, value in tmp.iteritems():
            if isinstance(value, int):
                self.numeric_col.append(index)
        return self.numeric_col

    def numericalData(self):
        if self.numeric_col is None:
            self.featureClassify()
        
        return Data(self[self.numeric_col], numeric_col=self.numeric_col, target=self.target)
    
    def categoricalData(self):
        if self.numeric_col is None:
            self.featureClassify()
        return Data(self.drop(columns=self.numeric_col), numeric_col=self.numeric_col, target=self.target)
    
# validation data set function    
    def holdOut(self, v_scale):
        msk = np.random.rand(len(self)) < v_scale
        return Data(self[msk], target=self.target, numeric_col=self.numeric_col), Data(self[~msk], target=self.target, numeric_col=self.numeric_col)
    
    def Kfold(self, k):
        dataset = []
        rand = int(self.shape[0]/k)

        for i in range(k):
            dataset.append(Data(self.sample(n=rand), target=self.target))
            self = self.drop(dataset[i].index)

        while not self.empty:
            for i in range(k):
                tmp = self.sample(n=1)
                dataset[i] = Data(dataset[i].append(tmp), target=self.target)
                self = self.drop(tmp.index)

                if self.empty:
                    break
        return dataset

    def genConfusion(self, predict):
        actual = pd.Series(self[self.target].tolist(), name='actual')
        predict = pd.Series(predict, name='predict')
        return pd.crosstab(predict,actual)
