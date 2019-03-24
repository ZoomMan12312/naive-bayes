from naivebayes import NB
import pandas as pd

data = pd.read_csv('data.csv')
y = data['Index']
x_con = data[['Weight','Height']]

nb = NB()

nb.fit(x_con,1,y)
score = nb.score(x_con,1,y)
scoreint = nb.scoreInterval(x_con,1,y,10)
print(scoreint)
plot_false = []
plot_total = []
names = []

import matplotlib.pyplot as plt
for i in scoreint:
    plot_false.append(i[2])
    plot_total.append(i[3])
    names.append(str(int(i[0]*100)) + '-' + str(int(i[1]*100)))

#x = range(len(plot_false))

plt.bar(names, plot_false)
plt.plot(plot_total, 'r--')
plt.plot(plot_total, 'ro')
plt.show()
'''
pred = nb.predict([80, 173], 0)
print(nb.y_uniq)
#probas = nb.getProbas([180, 70], 0)
#nb.showGaussianDist(1,0)
#input()

from matplotlib import cm
import numpy as np
#arr = np.array([[0,0,0,0,0]])
#arr = np.append(arr, [[1,0,1,1,1]], axis=0)
#print(arr)
arr = []
for x in range(100,200):
    row = []
    for y in range(50,200):
        pred = nb.predict([y, x], 0)
        #(x:weight, y:height)
        row.append(pred)
    arr.append(row)
print(nb.y_uniq)
narr = np.array(arr)
pyplot.imshow(narr, interpolation='nearest', cmap=cm.Blues)
pyplot.show()
'''