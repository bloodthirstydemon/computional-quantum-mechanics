# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:18:57 2024

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

with open('Parameters for ex_5.txt', 'r') as file:
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



# Constants
m = float(params.get('m'))
C = float(params.get('C'))
D = float(params.get('D'))
E = float(params.get('E'))


# Numerical parameters
x_min = float(params.get('x_min'))
x_max = float(params.get('x_max'))
num_points = float(params.get('num_points'))
dx = (x_max - x_min) / (num_points - 1)


################################################################################################################
'Potential function'
################################################################################################################

def potential(x):
    return 0.5 * C * x**2 - 0.5 * D * x**3 + 0.5 * E * x**4

################################################################################################################
'Numerov method solver'
################################################################################################################


def numerov_method(potential, energy, num_points, dx):
    x = np.linspace(x_min, x_max, int(num_points))
    psi = np.zeros(int(num_points))
    k_sq = 2.0 * m * (potential(x) - energy)

    # Initial values
    psi[0] = 0.0
    psi[1] = 1e-6

    # Numerov algorithm
    for i in range(2, int(num_points)):
        factor = 1.0 + (dx**2 / 12.0) * k_sq[i - 1]
        psi[i] = (2 * psi[i - 1] * (1 - 5 * (dx**2 / 12.0) * k_sq[i - 1]) -
                  psi[i - 2] * (1 + (dx**2 / 12.0) * k_sq[i - 2])) / factor

    # Normalize the wavefunction
    normalization = np.sqrt(np.trapz(np.abs(psi)**2, x))
    psi_normalized = psi / normalization

    return x, psi_normalized


################################################################################################################
'Find eigenstates and eigenenergies'
################################################################################################################


num_eigenstates = 5
eigenstates = []
eigenenergies = []

for n in range(num_eigenstates):
    # Initial guess for energy (you may need to adjust this based on your problem)
    energy_guess = C * (n + 0.5)
    
    # Solve using Numerov method
    x, psi = numerov_method(potential, energy_guess, num_points, dx)
    
    # Store eigenstates and eigenenergies
    eigenstates.append(psi)
    eigenenergies.append(energy_guess)

    # Plot eigenstates
    plt.plot(x, psi, label=f"Energy = {energy_guess:.2f}")

plt.title("Eigenstates of the Anharmonic Oscillator (Normalized)")
plt.xlabel("x")
plt.ylabel("Normalized Wave Function")
plt.legend()
plt.show()

# Print eigenenergies
print("Eigenenergies:")
for i, energy in enumerate(eigenenergies):
    print(f"Eigenenergy {i + 1}: {energy:.4f}")
