"""
@author: Johannes Wiesel 

Plots the Pearson correlation coeefficent, Distance correlation coeefficent, 
(unnormalised) Hellinger correlation coeefficent and Wasserstein correlation coefficient
for the bivariate standard normal case as a function of the correlation rho
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

eps = 0.001 
rho = np.linspace(0, 1-eps, num=1000)
dcov = ((rho*np.arcsin(rho)+np.sqrt(1-rho**2) - rho*np.arcsin(rho/2)-np.sqrt(4-rho**2)+1)/
        (1+np.pi/3-np.sqrt(3)))
t = 1 - np.sqrt(1-rho**2)
hel = 1 - (2*(1-rho**2)**(1/4))/(4-rho**2)**(1/2)

f = plt.figure(figsize=(11.69,8.27))
plt.plot(rho,rho, label= "Pearson's correlation")
plt.plot(rho, dcov, label = "Distance correlation")
plt.plot(rho, hel, label = "Hellinger correlation")
plt.plot(rho, t, label = "Wasserstein correlation")
plt.legend()
plt.show()
