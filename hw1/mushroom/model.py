import math
import numpy as np
import pandas as pd

def naive_bayes_classifier(freq_tables, test):
    #li = []
    pos = 0
    neg = 0
    predictE = math.log2(freq_tables[0].e)
    predictP = math.log2(freq_tables[0].p)

    for index in test.index:
        probE = predictE
        probP = predictP

        for col in test.columns:
            if col is 0:
                continue

            table = freq_tables[col]
            feature = test[col][index]

            if not np.isnan(table['e'][feature]):
                # print("{},e,{}: {}".format(col, feature, table['e'][feature]))
                probE += math.log2(table['e'][feature])
            if not np.isnan(table['p'][feature]):
                # print("{},p,{}: {}".format(col, feature, table['p'][feature]))
                probP += math.log2(table['p'][feature])
        #li.append(int(np.argmax([probE, probP])))

        if int(np.argmax([probE, probP])) is 0 and test[0][index] is 'e':
            pos += 1 
        elif int(np.argmax([probE, probP])) is 1 and test[0][index] is 'p': 
            pos += 1
        else:
            neg += 1
        # print(probE)
        # print(probP)
        # print()
    #ans = pd.DataFrame(li).groupby([0]).size()
    #print(ans)
    #print()
    #print(test.groupby([0]).size())
    total = test.groupby([0]).size()
    print()
    print("accuracy: {0:.0%}".format(pos/total.sum()))
