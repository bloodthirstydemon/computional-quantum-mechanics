# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:35:09 2023

@author: akbar
"""

#computational physics 
#Homework 1(forward euler mathod)

#import packages

import numpy as np
import matplotlib.pyplot as plt


#  i dψ(t)/dt = Hψ(t) solve this differencial equation using euler method


params = {}

with open('parameter(EX1).txt', 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue
        key, value = line.strip().split('=')
        key = key.strip()
        value = value.strip()
        params[key] = value
        
#calling all the parameters form parameters file, including initial conditions and assigning a variable to it

E0 = float(params.get('E0'))
total_time = float(params.get('total_time'))
time_step = float(params.get('time_step'))
t = float(params.get('t'))
psi = np.complex64(params.get('psi'))

num_steps = int(total_time / time_step)


# Lists to store data for plotting
time_values = []
psi_values = []

# Euler method implimentation

# Initialize time and wave function

for step in range(num_steps):
    t += time_step
    psi = -1.0j * E0 * psi * time_step + psi     #question:why does order of psi matter here? if we put it in beginning code does not run!
    time_values.append(t)
    psi_values.append(psi)

#plotting the results of wavefunction evolution
plt.plot(time_values, np.real(psi_values), label='Re(psi(t))')
plt.plot(time_values, np.imag(psi_values), label='im(psi(t))')
plt.title(f'for dt={time_step}(fs), E0={E0}(eV), psi = 1+0j')
plt.xlabel('Time(fs)')
plt.ylabel('psi(t)')
plt.legend()
plt.show()