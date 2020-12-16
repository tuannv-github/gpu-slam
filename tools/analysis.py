import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

DATA_DIR_GPU = "GPU"
DATA_DIR_CPU = "CPU"

xss = []
yss = []
xs = []
ys = []

for filename in os.listdir(DATA_DIR_GPU):
    print(filename)

    x = float(filename)-1
    f = open(DATA_DIR_GPU + "/" + filename, 'r')
    y = 0
    for line in f.readlines():
        yt = float(line)
        print(str(x) + ": " + str(y))
        if(y == 0):
            y =  yt
        else:
            y = (y + yt)/2
    xs.append(x)
    ys.append(y)
plt.bar(xs,ys, width = 2, color = 'b', label ='GPU')

xs = []
ys = []
for filename in os.listdir(DATA_DIR_CPU):
    print(filename)
    x = float(filename)+1
    f = open(DATA_DIR_CPU + "/" + filename, 'r')
    y = 0
    for line in f.readlines():
        yt = float(line)
        print(str(x) + ": " + str(y))
        if(y == 0):
            y =  yt
        else:
            y = (y + yt)/2
    xs.append(x)
    ys.append(y/10)

plt.bar(xs,ys, width = 2, color = 'g', label ='CPU/10')

plt.xlabel('Particles', fontweight ='bold') 
plt.ylabel('Micro Second', fontweight ='bold')
plt.legend()
plt.show()