from LogisticRegression import LogisticRegression as Logistic
import pandas as pd



if __name__ == "__main__":
    no = int(input('data set no: '))
    if no is 1:
        data1 = pd.read_csv('data/Logistic_data1-1.txt', header=None)
        data2 = pd.read_csv('data/Logistic_data1-2.txt', header=None)
    elif no is 2:
        data1 = pd.read_csv('data/Logistic_data2-1.txt', header=None)
        data2 = pd.read_csv('data/Logistic_data2-2.txt', header=None)

    data1.columns = ['x', 'y']
    data2.columns = ['x', 'y']

    logistic = Logistic(data1=data1, data2=data2)
    # logistic.plot()

    try:
        no = int(input('with error function[0: L2-norm, 1: cross entropy]: '))
        if no is 0:
            logistic.fitByL2Norm()
        else:
            logistic.fitbyCrossEntropy()
    except KeyboardInterrupt:
        pass

    p0 ,p1 = logistic.predict(data1.append(data2))
    data = pd.DataFrame()
    data['actual'] = [0]*data1.shape[0] + [1]*data2.shape[0]
    tmp = p0.append(p1)
    tmp[tmp['predict'] > 0.5] = int(1)
    tmp[tmp['predict'] <= 0.5] = int(0)
    data = data.join(tmp['predict'].astype('int64'))
    confusion_matrix = pd.crosstab(data['predict'], data['actual'])
    print()
    print('confussion matrix: ')
    print(confusion_matrix)

    print(confusion_matrix[0][1])
    print('precision: ' + str(confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])))
    print('recall: ' + str(confusion_matrix[0][0]/confusion_matrix[0].sum()))

    logistic.printWeight()
    logistic.plot(p0, p1)
