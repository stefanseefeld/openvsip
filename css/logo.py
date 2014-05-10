#! /usr/bin/env python

import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

def s(t):
    return erfc(t)/2

def o(t, d = 0):
    return np.exp(-(t+d)*(t+d)*0.6)


def i(t, d=0):
    return np.sin(np.pi*(t+np.pi-0.3)+d)
def f(t, d=0):
    return o(t,-.5) * i(t,d) + s(t)


t = np.arange(-5.0, 5.0, 0.02)
plt.figure(1, figsize=(15,5))
plt.plot(t, f(t,-2), color = '#eeefff')
plt.plot(t, f(t,-1), color = '#eeefff')
plt.plot(t, f(t,-0.5), color = '#eeefff')
plt.plot(t, f(t,-0.25), color = '#eeefff')
plt.plot(t, f(t,0.25), color = '#eeefff')
plt.plot(t, f(t,0.5), color = '#eeefff')
plt.plot(t, f(t,1), color = '#eeefff')
plt.plot(t, f(t,2), color = '#eeefff')
plt.plot(t, f(t,0), color = '#0000ff')

plt.show()
