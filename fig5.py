"""
@author: Johannes Wiesel 

Plots the Power of tests for independence defined in Cor. 5.2, 
Theorem 2.1 of Chatterjee (2020), 
Theorem 6 of Szekely, Rizzo, Bakirov (2007) and
Theorem 3.1 of Shi, Drton and Han
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import  norm
from scipy.spatial.distance import pdist, squareform
from xicor.xicor import Xi
import ot
from statsmodels.distributions.empirical_distribution import ECDF

def distcov_test(X, Y):
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
    S = np.abs(X-X.reshape((1,n))).sum()/n**2 * np.abs(Y-Y.reshape((1,n))).sum()/n**2
    return n*dcov2_xy/S

def distcov_test_norm(X, Y):
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
    S = np.abs(X-X.reshape((1,n))).sum()/n**2 * np.abs(Y-Y.reshape((1,n))).sum()/n**2
    return n*dcov2_xy

def adapW1_eot(x,y,N):
    x_new = np.floor(np.sqrt(np.log(N))*x)/np.sqrt(np.log(N))
    y_new = np.floor(np.sqrt(np.log(N))*y)/np.sqrt(np.log(N))

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
    return np.dot(W, x_freq)/N

def func(x):
    return x

rho = np.linspace(-0.5,0.5, num=50)
N = 1500
M = 200
alpha = 0.1
indW = np.zeros(M)
indD = np.zeros(M)
indC = np.zeros(M)
indXi = np.zeros(M)
freqW =  np.zeros(len(rho))
freqD =  np.zeros(len(rho))
freqC = np.zeros(len(rho))
freqXi = np.zeros(len(rho))

for i in range(1, len(rho)):
    for j in range(1, M):
        print(i,j)
        x = np.random.random_sample(N)
        y = func(rho[i]*x + np.sqrt(1-rho[i]**2)*np.random.random_sample(N))
        test = adapW1_eot(x,y,N)
        indW[j] = (test <= np.sqrt(np.log(N))*np.sqrt(2/np.pi/N) + (1-2/np.pi)/np.sqrt(N)*norm.ppf(1-alpha))
        indD[j] = (distcov_test(x,y) <= norm.ppf(1-alpha/2)**2)
        ecdf_x = ECDF(x)
        ecdf_y = ECDF(y)
        indC[j] = (distcov_test_norm(ecdf_x(x),ecdf_y(y)) <= 0.306)
        indXi[j] = (np.sqrt(N)*Xi(x,y).correlation <= norm.ppf(1-alpha/2)*np.sqrt(2/5))
    freqW[i] =  np.mean(indW)
    freqD[i] = np.mean(indD)
    freqC[i] = np.mean(indC)
    freqXi[i] = np.mean(indXi)
        

f = plt.figure(figsize=(11.69,8.27))   
plt.plot(rho, 1 - freqW , label= "Power of the Wasserstein test")
plt.plot(rho, 1 - freqD , label= "Power of the Distance covariance test")
plt.plot(rho, 1 - freqC , label= "Power of the Center-outward covariance test")
plt.plot(rho, 1 - freqXi , label= "Power of Chatterjee correlation test")
plt.legend()
plt.show()

