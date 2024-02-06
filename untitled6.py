# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:40:43 2024

@author: akbar
"""
#################################################################################################################
'Importing packages'
#################################################################################################################


import numpy as np
from scipy import constants
import scipy
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


#################################################################################################################
'loading parameter file'
#################################################################################################################


params = {}

with open('New Text Document.txt', 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue
        key, value = line.strip().split('=')
        key = key.strip()
        value = value.strip()
        params[key] = value


################################################################################################################
'calling all the parameters form parameters file, including initial conditions and assigning a variable to it'
################################################################################################################

femto = constants.femto
E_if = (float(params.get('E_if')))
I0 = (float(params.get('I0')))
T = (float(params.get('T')))*(femto/constants.value('atomic unit of time'))
t0 = (float(params.get('t0')))*(femto/constants.value('atomic unit of time'))
t_final = (float(params.get('t_final')))*(femto/constants.value('atomic unit of time'))
num_points = (int(params.get('num_points')))
omega = (float(params.get('omega')))
t_points = np.linspace(t0, t_final, num_points)
dt = t_points[1] - t_points[0]


################################################################################################################
'definition of gaussian integration function'
################################################################################################################

def gauss_legendre_integration(func, a, b):
    # Two-point Gauss-Legendre Quadrature
    nodes = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1, 1])

    # Map nodes to the integration interval [a, b]
    mapped_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)

    result = 0.5 * (b - a) * np.sum(weights * func(mapped_nodes))

    return result


################################################################################################################
'definition of integral we want to evaluate which includes pulse shape'
################################################################################################################

# =============================================================================
# 
# def integrand(t):
#     
#     pulse_width = T 
#     for i in t:
#         if abs(i) < pulse_width/2:
#             function = np.exp(-1j * (omega + E_if) * i)
#         else:
#             function = 0
#     
#     
#     return function #np.exp(-t**2 / T**2) * np.exp(-1j * (omega + E_if) * t)
# 
# =============================================================================

def integrand(t):
    
    return np.exp(-t**2 / T**2) * np.exp(-1j * (omega + E_if) * t)




################################################################################################################
'calculation and  plotting'
################################################################################################################


occupa_prob = []
result_plot = []
result = 0

for i in t_points:
    a = i
    b = i+dt
    result += (gauss_legendre_integration(integrand, a, b))
    result_plot.append(result)
    #occupa_prob.append(np.abs(gauss_legendre_integration(integrand, a, b))**2)


plt.plot(t_points/(femto/constants.value('atomic unit of time')), I0*(np.abs(result_plot)**2)/max(I0*(np.abs(result_plot)**2)), label=f'Omega = {omega}\n final probability = {I0*(np.abs(result)**2)/max(I0*(np.abs(result_plot)**2))}')
plt.xlabel('Time (fs)')
plt.ylabel('Occupation Probability')
plt.title('Occupation Probability of State $|f\\rangle$ vs Time')
plt.legend()
plt.show()

print("Result:", I0*(np.abs(result)**2)/max(I0*(np.abs(result_plot)**2)))
