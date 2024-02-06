# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:13:39 2024

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

with open('# Parameters.txt', 'r') as file:
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


N = (int(params.get('N')))
J = (float(params.get('J')))
B_lower = (float(params.get('B_lower')))
B_higher = (float(params.get('B_higher')))
B_values = np.linspace(B_lower, B_higher, 21)
T_lower = (float(params.get('T_lower')))
T_higher = (float(params.get('T_higher')))
temperature_values = np.linspace(T_lower, T_higher, 50)
num_steps = (int(params.get('num_steps')))
discard_steps = (int(params.get('discard_steps')))
fixed_B = (float(params.get('fixed_B')))
fixed_temperature = (float(params.get('fixed_temperature')))


# Function to calculate energy of a given configuration
def calculate_energy(spins, J, B):
    N = len(spins)
    return -J * np.sum(spins * np.roll(spins, 1)) - B * np.sum(spins)

################################################################################################################
'perform Metropolis algorithm'
################################################################################################################


def metropolis(spins, beta, J, B):
    N = len(spins)
    for _ in range(N):
        i = np.random.randint(N)  # Choose a random spin
        delta_E = 2 * J * spins[i] * (spins[(i + 1) % N] + spins[(i - 1) % N]) + 2 * B * spins[i]

        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i] *= -1  # Flip the spin with a certain probability

################################################################################################################
'simulate_ising_model'
################################################################################################################


def simulate_ising_model(N, J, B, beta_values, num_steps, discard_steps):
    magnetization_data = []

    for beta in beta_values:
        spins = np.random.choice([-1, 1], size=N)  # Initial random spin configuration

        for step in range(num_steps + discard_steps):
            metropolis(spins, beta, J, B)

            # Discard initial steps to reach equilibrium
            if step >= discard_steps:
                magnetization_data.append(np.sum(spins) / N)

    average_magnetization = np.mean(magnetization_data)
    return average_magnetization


################################################################################################################
'calculating and ploting the results'
################################################################################################################

#magnetization as a function of temperature for a fixed megnetic field

beta_values_fixed_B = 1 / temperature_values
magnetization_temperature_fixed_B = [simulate_ising_model(N, J, fixed_B, [beta], num_steps, discard_steps) for beta in beta_values_fixed_B]

# Plot magnetization as a function of megnetic field for a fixed temperature
beta_fixed_temperature = 1 / fixed_temperature
magnetization_B_fixed_temperature = [simulate_ising_model(N, J, B, [beta_fixed_temperature], num_steps, discard_steps) for B in B_values]


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(temperature_values, magnetization_temperature_fixed_B, marker='o')
plt.xlabel('Temperature (K)')
plt.ylabel('Average Magnetization')
plt.title(f'Magnetization vs. Temperature {fixed_B} in a.u.')

plt.subplot(1, 2, 2)
plt.plot(B_values, magnetization_B_fixed_temperature, marker='o')
plt.xlabel('External Magnetic Field (B)')
plt.ylabel('Average Magnetization')
plt.title('Magnetization vs. External Magnetic Field {fixed_temperature} in K')

plt.tight_layout()
plt.show()