import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression as Linear

if __name__ == "__main__":
    data = pd.read_csv("data/linear_data.txt", header=None)
    data.columns = ['x', 'y']

    n = int(input('n = '))
    linear = Linear(data=data)
    linear.fitLine(n)
    linear.printLine()
    print('Total error: ' + str(linear.totalError()))
    linear.plot(line=True)

   