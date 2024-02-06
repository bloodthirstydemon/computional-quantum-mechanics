# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:33:59 2023

@author: akbar
"""

#################################################################################################################
'Importing packages'
#################################################################################################################

from scipy.integrate import simps
import numpy as np
from scipy import constants
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from qutip import (about, basis, coherent, coherent_dm, displace, fock, ket2dm,
                   plot_wigner, squeeze, thermal_dm, Qobj, wigner)


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

e = constants.e
J = constants.value('atomic unit of energy')
femto = constants.femto
Ang = constants.angstrom

T = float(params.get('T'))*(femto/constants.value('atomic unit of time'))
x = float(params.get('x'))*(Ang/constants.value('atomic unit of length'))
σ = float(params.get('sigma'))
k = float(params.get('k'))*constants.value('atomic unit of length')/Ang
m = float(params.get('m'))
w = float(params.get('w'))*constants.value('atomic unit of length')/Ang
v = float(params.get('v'))*e/J
l = float(params.get('l'))*constants.value('atomic unit of length')/Ang
K = float(params.get('K'))
potential_well_width = float(params.get('potential_well_width'))*constants.value('atomic unit of length')/Ang
height = float(params.get('height'))*e/J


number_of_steps_x = int(params.get('number_of_steps_x'))
number_of_steps_t = int(params.get('number_of_steps_t'))
nt = (np.linspace(0, T, int(number_of_steps_t)))
nx = (np.linspace(-x, x, int(number_of_steps_x)))
dt = T/len(nt)
dx = (x+x)/len(nx)


################################################################################################################
'defining function for initial wavefunction and normalise'
################################################################################################################


def psi0(nx):
    
    D = 1/((σ*((2*np.pi)**(1/2)))**(1/2))
    e1 = np.exp(1j*k*nx)
    e2 = np.exp(-((nx**2)/4*(σ**2)))
    return (D)*e1*e2


def normalize_wavefunction(psi, dx):
    # Calculate the normalization constant
    normalization_constant = np.sqrt(np.trapz(np.abs(psi)**2, dx=dx))

    # Normalize the wavefunction
    normalized_psi = psi / normalization_constant

    return normalized_psi


################################################################################################################
'defining infinite_potential_well and step_potential'
################################################################################################################


def V(nx, height, v, potential_well_width, w, K):
    well_potential = height * (nx >= potential_well_width / 2 or nx <= -potential_well_width / 2)
    harmonic_potential = (1/2)*K*((nx)**2)
    #step_potential = v * (l - (w / 2) <= nx) * (nx <= (l + w / 2))
    return well_potential #+ harmonic_potential #+ step_potential


################################################################################################################
'Wigner_function implementation'
################################################################################################################


def wigner_function(psi, x, p):
    """
    Calculate the Wigner function using QuTiP.

    Parameters:
    - psi (array): Wavefunction in position space.
    - x (array): Position values.
    - p (array): Momentum values.
    - h_bar (float): Reduced Planck constant (default is 1.0).

    Returns:
    - W (array): Wigner function.
    """
  
    
    # Create position and momentum operators

    # Calculate the Wigner function
    psi = Qobj(psi)
    x = Qobj(x)
    p = Qobj(p)
    W = wigner(psi, x, p)
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
psin0 = normalize_wavefunction(psi0(nx), dx)

for i in range(number_of_steps_t):
    psin1 = solve(A, (B @ psin0))# Crank-Nicolson algorithm
    psi_values.append(psin1)
    psin0 = psin1
    
    
################################################################################################################
'Save frames as PNG images'
################################################################################################################
P_values = np.linspace(-20, 20, len(nx))#np.linspace(0, max(np.fft.fftshift(np.fft.fftfreq(len(nx), d=dx)))*2, len(nx))
    

for frame, psi in enumerate(psi_values):
    #Calculate and store the Wigner function 
    #plot_wigner_2d_3d(psi)
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nx, np.abs(psi)**2)
    plt.ylim([0, .40])
    plt.axvspan(-potential_well_width / 2, potential_well_width / 2, alpha=0.2, label='Potential Well')
    #plt.axvspan(l-w*dx,l+w*dx, ymin = 0, ymax = v/4, alpha=0.2,color = 'red', label='step Potential')
    plt.xlabel('Position (10^(-10)m)')
    plt.ylabel('Probability (|Ψ(x,t)|^2)(normalised)')
    plt.title('Wavefunction and Potential Well')
    plt.legend(loc = 9)
    plt.grid(True)
    plt.title(f'Time Evolution - {round(((T/len(nt))*frame), 5)}fs')
    wigners = np.abs(wigner_function(psi, nx, P_values))
    ax = fig.add_subplot(1, 2, 2)
    plt.contourf(nx, P_values, wigners, levels=500, cmap='viridis')
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
