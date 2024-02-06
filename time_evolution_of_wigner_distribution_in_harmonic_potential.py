# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:16:40 2023

@author: akbar
"""

#################################################################################################################
'Importing packages'
#################################################################################################################

import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

#################################################################################################################
'loading parameter file'
#################################################################################################################

params = {}

with open('Parameter_for_time dependent_wigner function of gaussian packet.txt', 'r') as file:
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

T = float(params.get('T'))
x = float(params.get('x'))
σ = float(params.get('sigma'))
k = float(params.get('k'))
m = float(params.get('m'))
w = float(params.get('w'))
v = float(params.get('v'))
l = float(params.get('l'))
K = float(params.get('K'))
potential_well_width = float(params.get('potential_well_width'))
height = float(params.get('height'))
number_of_steps_x = int(params.get('number_of_steps_x'))
number_of_steps_t = int(params.get('number_of_steps_t'))
nt = (np.linspace(0, T, int(number_of_steps_t)))
nx = (np.linspace(-x, x, int(number_of_steps_x)))
dt = T/len(nt)
dx = (x+x)/len(nx)

################################################################################################################
'defining function for initial wavefunction and normalise'
################################################################################################################

'''def psi0(nx, x0 = 0, sigma = 1, p0 = 10, h_bar=1.0):
    return np.exp(-(nx - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * nx / h_bar)'''

def psi0(nx):
    
    D = 1/((σ*((2*np.pi)**(1/2)))**(1/2))
    e1 = np.exp(1j*k*nx)
    e2 = np.exp(-((nx**2)/4*(σ**2)))
    return (D)*e1*e2

'''def normalize_wavefunction(psi):
    norm_factor = np.sqrt(np.sum(np.abs(psi)**2))
    normalized_psi = psi / norm_factor
    return normalized_psi'''

################################################################################################################
'defining infinite_potential_well and step_potential'
################################################################################################################

def V(nx, height, v, potential_well_width, w, K):
    well_potential = height * (nx >= potential_well_width / 2 or nx <= -potential_well_width / 2)
    harmonic_potential = (1/2)*K*((nx)**2)
    #step_potential = v * (l - (w / 2) <= nx) * (nx <= (l + w / 2))
    return well_potential + harmonic_potential #+ step_potential

################################################################################################################
'Wigner_function implementation'
################################################################################################################

def wigner_function(psi, x, p, h_bar=1.0):
    W = np.zeros((len(x), len(p)), dtype=complex)
    dx = x[1] - x[0]

    for i in range(len(x)):
        for j in range(len(p)):
            W[i, j] = (1.0 / (np.pi * h_bar)) * np.trapz(
                       np.conj(psi) * np.roll(psi, i) * np.exp(-1j * p[j] * x / h_bar), x)

    return W

################################################################################################################
'defining tri-diagonal matrix'
################################################################################################################

#construct Alpha & Bata

tau = dt
h = dx
D = (1j)/2*m

alpha_values = []
beta_values = []

for i in nx:
        alpha = (2*(dx**2))/(tau*D) + 2 + ((1j*(dx**2)*V(i, height, v, potential_well_width, w, K))/D)
        alpha_values.append(alpha)
        beta = (2*(dx**2))/(tau*D) - 2 - ((1j*(dx**2)*V(i, height, v, potential_well_width, w, K))/D)
        beta_values.append(beta)
    
# Construct tridiagonal matrix A
A_upper = -1 * np.ones((len(nx)-1))
A_lower = -1 * np.ones(len(nx) - 1)
A = np.diag(alpha_values) + np.diag(A_upper, 1) + np.diag(A_lower, -1)

# Construct tridiagonal matrix B
B_upper = 1 * np.ones(len(nx) - 1)
B_lower = 1 * np.ones(len(nx) - 1)
B = np.diag(beta_values) + np.diag(B_upper, 1) + np.diag(B_lower, -1)

################################################################################################################
'solving time dependent schrödinger equation by Crank_Nicolson algorithm implimentation'
################################################################################################################

psi_values = []
psin0 = (psi0(nx))

for i in range(number_of_steps_t):
    psin1 = solve(A, (B @ psin0))# Crank-Nicolson algorithm
    psi_values.append(psin1)
    psin0 = psin1

################################################################################################################
'Save frames as PNG images'
################################################################################################################

for frame, psi in enumerate(psi_values):
    P_range = 40
    P_values = np.linspace(-P_range/2, P_range/2, len(nx))
    # Calculate and store the Wigner function 
    wigner = np.abs(wigner_function(psi, nx, P_values))
    plt.contourf(nx, P_values, wigner, levels=500, cmap='viridis')
    plt.title('Wigner Distribution')
    plt.xlabel('Position (x)')
    plt.ylabel('Momentum (p)')
    plt.colorbar(label='Absolute magnitude')
    plt.grid(True)
    plt.title(f'Time Evolution - {round(((T/len(nt))*frame), 5)}fs')

    # Create a folder to save the figures
    output_folder = 'C:/Users/akbar/Downloads/pictures_2'
    os.makedirs(output_folder, exist_ok=True)
    # Save the plot in the specified folder
    output_filename = os.path.join(output_folder, f'frame_{frame:03d}.png')
    plt.savefig(output_filename)
    plt.close()
