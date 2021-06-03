"""
@author: Johannes Wiesel 

Prints the Pearson correlation coeefficent, Spearman correlation coeefficent,
Distance correlation coeefficent, Chatterjee's correlation coefficient
and Wasserstein correlation coefficient for Galton's peas data
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import pdist, squareform
from xicor.xicor import Xi
import pickle 
import ot

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def adapW1_eot(x,y,N):
    #Assume d=1 and thus quantisation N^(-1/3)
    
    x_new = N**(-1/3)*np.floor(N**(1/3)*x)
    y_new =  N**(-1/3)*np.floor(N**(1/3)*y)
    #plt.plot(x_new, y_new, '.')

    x_val = np.array(list(Counter(x_new).keys()))
    x_freq = np.array(list(Counter(x_new).values()))
    W = np.zeros(len(x_val))
    for i in range(0,len(x_val)):
        aux = y_new[x_new==x_val[i]]
        aux = aux.reshape((len(aux), 1))
        c = np.abs(aux-y_new)
        w1 = np.ones(len(aux))/len(aux)
        w2 = np.ones(len(y))/len(y)
        W[i] = ot.sinkhorn2(w1,w2,c,0.01)
    c = np.abs(y_new.reshape((N,1))-y_new)
    denom = c.sum()/N**2
    return np.dot(W, x_freq)/(N*denom)


peas = []
import csv
with open('peas.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        peas.append([float(i) for i in row])
        
peas = np.array(peas)[:,1:3]

xi =  np.zeros(10**4)

adapW1_eot(peas[:,0], peas[:,1],len(peas[:,0]))
pearsonr(peas[:,0], peas[:,1])
spearmanr(peas[:,0], peas[:,1])
distcorr(peas[:,0], peas[:,1])
#Average over Chatterjee's coefficient as done in Chatterjee (2020)
for i in range(1,10**4):
    print(i)
    xi[i]=Xi(peas[:,0], peas[:,1]).correlation
np.mean(xi)


xi =  np.zeros(10**4)

adapW1_eot(peas[:,1], peas[:,0],len(peas[:,0]))
pearsonr(peas[:,1], peas[:,0])
distcorr(peas[:,1], peas[:,0])
#Average over Chatterjee's coefficient as done in Chatterjee (2020)
for i in range(1,10**4):
    print(i)
    xi[i]=Xi(peas[:,1], peas[:,0]).correlation
np.mean(xi)
spearmanr(peas[:,1], peas[:,0])
