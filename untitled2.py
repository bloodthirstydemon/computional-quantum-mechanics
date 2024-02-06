
#################################################################################################################
'Importing packages'
#################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

#################################################################################################################
'loading parameter file'
#################################################################################################################

params = {}

with open('parameter(EX2).txt', 'r') as file:
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

E1 = float(params.get('E1'))*1.6e-19
E2 = float(params.get('E2'))*1.6e-19
V0 = float(params.get('V0'))*1.6e-19
w = float(params.get('w'))*1e15
T = float(params.get('T'))*1e-15
dt = float(params.get('dt'))*1e-15
delta_E = abs(E1-E2)
h_bar = constants.hbar

# Initial
psi = np.array([1+0j, 0+0j], dtype=complex)

################################################################################################################
'defining Hamiltonian for two level system'
################################################################################################################

#Hamiltonian
def H(t):
    return np.array([[E1, V0 * np.cos(w * t)], [V0 * np.cos(w * t), E2]], dtype=complex)

################################################################################################################
'Runge-Kutta implementation'
################################################################################################################

t_points = np.arange(0, T, dt)
occupation_1 = []
occupation_2 = []

for t in t_points:
    k1 = (-1j /h_bar)* H(t) @ psi
    k2 = (-1j /h_bar)* H(t + dt / 2) @ (psi + k1 * dt / 2)
    k3 = (-1j /h_bar)* H(t + dt / 2) @ (psi + k2 * dt / 2)
    k4 = (-1j /h_bar)* H(t + dt) @ (psi + k3 * dt)
    psi += (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    occupation_1.append(np.abs(psi[0])**2)
    occupation_2.append(np.abs(psi[1])**2)
    
################################################################################################################
'analytica solution for v<<E1-E2'
################################################################################################################

def calculate_c1_squared(V, delta_E, t):
    term1 = 1 - (2 * (V ** 2) / (delta_E ** 2))
    term2 = 2 * ((V ** 2) / (delta_E ** 2)) * np.cos((delta_E + (2 * (V ** 2) / (delta_E))) * t)
    
    result = (term1 + term2)
    return result

################################################################################################################
'Ploting the results'
################################################################################################################

plt.figure(figsize=(10, 6))
plt.plot(t_points, occupation_1, label='Occupation |1>')
plt.plot(t_points, occupation_2, label='Occupation |2>')
plt.xlabel('Time (s)')
plt.ylabel('Occupation |ψ|2')
plt.title(f'Time evolution of Occupation for w={w/1e15}rad/fs, V={V0/1.6e-19}eV, E1={E1/1.6e-19}eV, E2={E2/1.6e-19}eV')
plt.legend()
plt.show()

