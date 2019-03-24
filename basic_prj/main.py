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
plot_err = []
plot_cml = []
plot_cml_err = []
names = []
total = 0
total_err = 0

for i in scoreint:
    total += i[3]
    total_err += i[2]

import matplotlib.pyplot as plt
cnt = 0
for i in scoreint:
    err = 0
    if i[2] and i[3] != 0:
        err = i[2]/i[3]
    plot_err.append(err)

    if cnt != 0:
        cm_to_add = i[3]/total + plot_cml[cnt-1]
        cm_to_add_err = i[2]/total_err + plot_cml_err[cnt-1]
    else:
        cm_to_add = i[3]/total
        cm_to_add_err = i[2]/total_err

    plot_cml_err.append(cm_to_add_err)
    plot_cml.append(cm_to_add)
    cnt += 1

    plot_false.append(i[2])
    plot_total.append(i[3])
    names.append(str(int(i[0]*100)) + '-' + str(int(i[1]*100)))

#x = range(len(plot_false))
plt.subplot(121)
plt.title('Err/Observations')
plt.bar(names, plot_false)
plt.plot(plot_total, 'g--')
plt.plot(plot_total, 'go')
plt.grid(True)
plt.subplot(122)
plt.title('(r)Err%/(g)Comulative Observations/(y)Comulative Errors')
plt.plot(names, plot_err, 'r--')
plt.plot(plot_err, 'ro')
plt.plot(plot_cml, 'g--')
plt.plot(plot_cml, 'go')
plt.plot(plot_cml_err, 'y--')
plt.plot(plot_cml_err, 'yo')
plt.grid(True)
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