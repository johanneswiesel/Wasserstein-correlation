"""
@author: Johannes Wiesel 
Plots a histogramm of the normalised empirical distribution of the Wasserstein correlation against a standard normal density
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  norm
import ot

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


# Independence asymptotic distribution test
N = 500
M = 10**4

W_test = np.zeros(M)

for i in range(1, M):
    print(i)
    x = np.random.random_sample(N)
    y = np.random.random_sample(N)
    W_test[i] = adapW1_eot(x,y,N)

x = np.linspace(-5,5)
f = plt.figure(figsize=(11.69,8.27))   
plt.hist(np.sqrt(N)*(W_test-np.sqrt(2*np.log(N)/np.pi/N)/6)/(1-2/np.pi), bins=50, density=True)
plt.plot(x, norm.pdf(x))
plt.show()

