import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import math

class NB():
    def __init__(self):
        pass
    
    def fit(self, x_con, x_cat, y): #y is a pandas frame with a single column
        self.y = y
        self.y_prior = self.getPrior()
        self.x_con = x_con
        self.x_con_cats = x_con.columns.values
        self.getMean()

    def getPrior(self):
        #sets priors for each category
        self.y_uniq = self.y.unique()
        self.y_uniq.sort()
        self.y_num = []
        for i in self.y_uniq:
            self.y_num.append(0)
        
        self.y_total = 0
        for i in self.y:
            cnt = 0
            for n in self.y_uniq:
                if i == n:
                    self.y_num[cnt] += 1
                    self.y_total += 1
                    break
                cnt += 1
        
        prior = []
        for i in self.y_num:
            prior.append(i/self.y_total)

        return prior

    def getMean(self):
        #sets mean and variance based on the training data
        totals = []
        means = []
        variance = []
        cnt = 0
        for i in self.y_uniq:
            totals.append([])
            means.append([])
            variance.append([])
            for n in self.x_con_cats:
                totals[cnt].append(0)
                variance[cnt].append(0)
            cnt += 1

        row = 0
        for i in self.x_con.values:
            cnt = 0
            y = self.getY(self.y.values[row])
            for n in i:
                totals[y][cnt] += n
                cnt+= 1
            row += 1

        cnt = 0
        for i in totals:
            for n in i:
                x = n/self.y_num[cnt]
                means[cnt].append(x)
            cnt += 1

        row = 0
        for i in self.x_con.values:
            cnt = 0
            y = self.getY(self.y.values[row])
            for n in i:
                variance[y][cnt] += ((n - means[y][cnt])**2)/self.y_num[y]
                cnt+= 1
            row += 1

        self.means = means
        self.variance = variance
        '''
        print(self.y_uniq)
        print(self.y_num)
        print(self.x_con_cats)
        print(means)
        print(variance)
        '''

    def getY(self, cat):
        cnt = 0
        for i in self.y_uniq:
            if i == cat:
                return cnt
            cnt += 1
    
    def getGaussProba(self, x, y, value):
        return math.exp(-((value - self.means[y][x])**2)/2/self.variance[y][x])/math.sqrt(2*math.pi*self.variance[y][x])

    def getProbas(self, x_num, x_cat):
        #print(self.y_uniq)
        probas = []
        probsum = 0
        for i in self.y_uniq:
            cat = self.getY(i)
            proba = 1
            feature = 0
            for n in x_num:
                proba *= self.getGaussProba(feature, cat, n)
                feature += 1
            probas.append(proba)
            probsum += proba

        for i in range(0, len(probas)):
            probas[i] /= probsum
        return probas #returns probabilities for each class

    def predict(self, x_num, x_cat): #RETURNS INDEX
        probas = self.getProbas(x_num, x_cat)
        maxproba = 0
        maxind = 0
        ind = 0
        for i in probas:
            if i > maxproba:
                maxind = ind
                maxproba = i
            ind += 1
        print('Class: ' + str(self.y_uniq[maxind]) + '(index:' + str(maxind) + ') probability: ' + str(maxproba*100) + "%")
        return [maxind, maxproba] #returns the index of the predicted category along with the probability of that category

    def score(self, x_num, x_cat, y):
        total = 0
        correct = 0
        row = 0
        for i in x_num.values:
            pred = self.predict(i,0)
            if y[row] == self.y_uniq[pred[0]]:
                correct += 1
                print('correct')
            else:
                print('false')
            total += 1
            row += 1
        return correct/total #returns accuracy % for the given sample

    def scoreInterval(self, x_num, x_cat, y, interval_amnt):
        #same as the scoring function, but on confidence intervals, set at interval_amnt
        #used to determine error % at different confidence levels
        #can be used with matplotlib to compare
        int_range = 1/interval_amnt
        intervals = []
        for i in range(0, interval_amnt):
            low = int_range * i
            high = int_range * (i + 1)
            intervals.append([low, high, 0, 0]) #(low, high, error, total)
        bcat = 0
        for b in intervals:
            total = 0
            false = 0
            row = 0
            for i in x_num.values:
                pred = self.predict(i,0)
                if pred[1] > b[0] and pred[1] <= b[1]:
                    if y[row] == self.y_uniq[pred[0]]:
                        
                        print('correct')
                    else:
                        false += 1
                        print('false')
                    total += 1
                row += 1
            intervals[bcat][2] = false
            intervals[bcat][3] = total
            bcat += 1
        return intervals #[interval_min, interval_max, errors, total]
    '''
    ####UNUSED
    def showGaussianDist(self, x, y):
        for i in range(50, 200):
            proba = self.getGaussProba(x, y, i)
            print('---')
            print(i)
            print(proba)

    def showAreas(self):
        e = Ellipse((self.means[0][0], self.means[0][1]), self.variance[0][0]/2, self.variance[0][1]/2)

        a = plt.subplot(111, aspect='equal')

        e.set_clip_box(a.bbox)
        e.set_alpha(0.1)
        a.add_artist(e)

        plt.show()
    '''