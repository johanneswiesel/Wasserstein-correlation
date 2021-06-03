"""
@author: Johannes Wiesel 

Plots the Pearson correlation coeefficent, Spearman correlation coeefficent,
Distance correlation coeefficent, Chatterjee's correlation coeefficent 
and Wasserstein correlation coefficient between (X_1, f(X_2)) 
for the bivariate uniform distribution (X_1, X_2) as a function of the
correlation rho for different functions f(x)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import pdist, squareform
from xicor.xicor import Xi
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
    x_new = N**(-1/3)*np.floor(N**(1/3)*x)
    y_new =  N**(-1/3)*np.floor(N**(1/3)*y)

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


N = 1000 #no. of samples
M = 30 #no. of draws

lam = np.linspace(0,1, num = 100)
Wcor = np.zeros(len(lam))
pcor = np.zeros(len(lam))
scor = np.zeros(len(lam))
dcor = np.zeros(len(lam))
ccor = np.zeros(len(lam))

Wcor_aux = np.zeros(M)
pcor_aux = np.zeros(M)
scor_aux = np.zeros(M)
dcor_aux = np.zeros(M)
ccor_aux = np.zeros(M)

def func(x):
    return np.abs(x-0.5)

for i in range(0,len(lam)):
    for j in range(0, M):
        print(i,j)
        x = np.random.random_sample(N)
        y = lam[i]*func(x)+(1-lam[i])*np.random.random_sample(N)
        Wcor_aux[j] = adapW1_eot(x,y,N)
        pcor_aux[j] , _ = pearsonr(x, y)
        dcor_aux[j] = distcorr(x, y)
        ccor_aux[j] = Xi(x,y).correlation
        scor_aux[j], _ = spearmanr(x,y)

    Wcor[i] = np.mean(Wcor_aux)
    pcor[i] = np.mean(pcor_aux)
    dcor[i] = np.mean(dcor_aux)
    ccor[i] = np.mean(ccor_aux)
    scor[i] = np.mean(scor_aux)

f = plt.figure(figsize=(11.69,8.27))
plt.plot(lam, Wcor, label="Wasserstein correlation")
plt.plot(lam, pcor, label="Pearson's correlation")
plt.plot(lam, scor, label="Spearman's correlation")
plt.plot(lam, dcor, label="Distance correlation")
plt.plot(lam, ccor, label="Chatterjee's correlation")
plt.legend()
plt.show()

