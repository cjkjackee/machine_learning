import pandas as pd
from decisionTree import randomForest
from MLdata import Data

def model_performance(confusion_matrix):
    tp = confusion_matrix[0][0]
    fn = confusion_matrix[0][1]
    fp = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = (tp)/(tp+fn)
    pre = (tp)/(tp+fp)
    return acc, sen, pre

if __name__ == "__main__":
    forest = randomForest()
    ########## read trainning data  ###########
    data = Data(pd.read_csv('data/X_train.csv', index_col='Id'))
    data.join(pd.read_csv('data/y_train.csv', index_col='Id'))
    data.setTarget('Category')

    data.dropMissing()
    data.shuffle()

    ############# read test data ############
    test = Data(pd.read_csv('data/X_test.csv', index_col='Id'))

    ##### hold out ##########
    validation, train  = data.holdOut(3/7)

    print('*'*10 + "hold out validation" + '*'*10)
    print("generate forest...")
    forest.gen(train)
    
    print('predicting...')
    result = forest.predict(validation)

    print('performance:')
    confusion_matrix = validation.genConfusion(result)
    print(confusion_matrix)
    acc, sen, pre = model_performance(confusion_matrix)
    print('accuracy: {}, sensitivity: {}, precision: {}'.format(acc, sen, pre))

    print()
    ############## K fold #############
    K = 3
    i = 1
    a_acc = 0
    a_sen = 0
    a_pre = 0
    print('*'*10 + "K fold validation(K={})".format(K) + '*'*10)
    for validation in data.Kfold(K):
        print('validation dataset {}'.format(i))
        train = data.drop(index=validation.index)
        
        print("generate forest...")
        forest.gen(train)
        
        print('predicting...')
        result = forest.predict(validation)

        print('perfromnce: ')
        confusion_matrix = validation.genConfusion(result)
        print(confusion_matrix)
        acc, sen, pre = model_performance(confusion_matrix)
        a_acc += acc
        a_sen += sen
        a_pre += pre
        i += 1
        print()
    print('accuracy: {}, sensitivity: {}, precision: {}'.format(a_acc/K, a_sen/K, a_pre/K))
    ############ result #############
    print('*'*10 + 'generate submition' + '*'*10)
    fd = open('randomForest.csv', 'w')
    print("generate tree...")
    forest.gen(data)
    
    print('predicting...')
    result = forest.predict(test)
    result = pd.DataFrame({data.target: result}, index=test.index)
    fd.write(result.to_csv())