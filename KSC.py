import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

datafilename = 'time_series_data.tsv'
resultfilename = 'result'
K = 5
MaxIter = 10

class Model():
    def __init__(self, datafilename):
        self.K = K
        self.MaxIter = MaxIter
        inputfile = open(datafilename, 'r')
        self.data = np.loadtxt(inputfile, delimiter = '\t', skiprows = 0)
        inputfile.close()
        self.N, self.L = self.data.shape
        self.data_normsquare = []
        for n in xrange(self.N):
            self.data_normsquare.append(np.linalg.norm(self.data[n],2) ** 2)
        self.mu = np.zeros([self.K, self.L])
        self.z = [random.randint(0, self.K-1) for _ in xrange(self.N)]
        self.updateC()       

    def updateC(self):
        self.C = defaultdict(list)
        for n in xrange(self.N):
            self.C[self.z[n]].append(n)
    
    def dhat(self, x, y):
        alpha = np.dot(x, y) / np.linalg.norm(y, 2)
        return np.linalg.norm(x - alpha * y, 2) / np.linalg.norm(x, 2)

    def dshift(self, x, mu_q):
        minval = 2
        for i in xrange(11):
            dhat = self.dhat(x, mu_q[i:self.L+i])
            if dhat < minval:
                minval = dhat
        return minval

    def getz(self, x):
        P = []
        for k in xrange(self.K):
            P.append(self.dshift(x, self.mu_q[k]))
        return P.index(min(P))    

    def E_step(self):
        for n in xrange(self.N):
            self.z[n] = self.getz(self.data[n])
        self.updateC()    

    def M_step(self):
        for k in xrange(self.K):
            M = np.zeros([self.L, self.L])
            for n in self.C[k]:
                M = M + np.identity(self.L) - np.dot(self.data[n].reshape(self.L, 1), self.data[n].reshape(1, self.L)) / self.data_normsquare[n]
            eigvalues, eigvectors = np.linalg.eig(M)
            print eigvalues
            self.mu[k] = eigvectors[:, np.argmin(eigvalues)]
            if sum(self.mu[k]) < 0:
                self.mu[k] = -self.mu[k]
        self.mu_q = np.column_stack((np.zeros([self.K, 5]), self.mu, np.zeros([self.K, 5])))

    def EM(self):
        for _ in xrange(self.MaxIter):
            self.M_step()
            self.E_step()

    def output(self, resultfilename):
        outputfile = open(resultfilename, 'w')
        fig = plt.figure(figsize = (self.K * 4, 3))
        for k in xrange(self.K):
            print >>outputfile, self.mu[k]
            fig.add_subplot(1, self.K, k + 1)
            plt.plot(self.mu[k])
        outputfile.close()
        plt.show()

    
if __name__ == '__main__':
    model = Model(datafilename)
    model.EM()
    model.output(resultfilename)
