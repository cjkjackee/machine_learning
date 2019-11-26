from MLdata import Data
import pandas as pd
import decisionTree as Dtree
import time

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
    ########## read trainning data  ###########
    data = Data(pd.read_csv('data/X_train.csv', index_col='Id'))
    data.join(pd.read_csv('data/y_train.csv', index_col='Id'))
    data.setTarget('Category')

    data.dropMissing()
    # data.fillMissing()
    data.shuffle()

    ############# read test data ############
    test = Data(pd.read_csv('data/X_test.csv', index_col='Id'))

    ##### hold out ##########
    validation, train  = data.holdOut(3/7)

    # t = time.process_time()
    print("#################### hold out validation ############################ ")
    print("generate tree...")
    root = Dtree.genTreeV2(train, 5)
    # print("time gen tree: {}".format(time.process_time() - t))
    
    print('predicting...')
    result = Dtree.predict(root, validation)

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
    print("######################## K fold validation(K={}) ###############################".format(K))
    for validation in data.Kfold(K):
        print('validation dataset {}'.format(i))
        train = data.drop(index=validation.index)
        
        print("generate tree...")
        root = Dtree.genTreeV2(train, 5)
        
        print('predicting...')
        result = Dtree.predict(root, validation)

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

    print()
    ############ result #############
    print('*'*10 + 'generate submition' + '*'*10)
    fd = open('result.csv', 'w')
    print("generate tree...")
    root = Dtree.genTreeV2(data, 5)
    
    print('predicting...')
    result = Dtree.predict(root, test)
    result = pd.DataFrame({data.target: result}, index=test.index)
    fd.write(result.to_csv())