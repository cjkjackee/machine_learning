import pandas as pd
import neuralNetwork as ann

if __name__ == "__main__":
    data = pd.read_csv('data.txt', header=None, names=['x', 'y', 'target'])
    actual = data['target']

    try:
        network = ann.load('network.data')
    except:
        network = ann.genNetwork(data.shape[1], 2)

    try:
        network.train(data,show=1)
    except KeyboardInterrupt:
        pass

    network.save('network.data')

    # plot actual data
    pos = data[data['target'] == 1]
    neg = data[data['target'] == 0]
    ann.plot(pos, neg)

    # predict data
    tmp = data.drop(columns=['target'])
    tmp['predict'] = network.predict(tmp)

    # plot predict data
    pos = tmp[tmp['predict'] == 1]
    neg = tmp[tmp['predict'] == 0]
    ann.plot(pos, neg)

    ann.accuracy(tmp['predict'], actual)
