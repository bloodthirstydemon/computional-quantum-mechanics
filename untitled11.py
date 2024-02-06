# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:39:43 2024

@author: akbar
"""
#################################################################################################################
'Homework_4(Split-Operator method to solve time dependent schrÃ¶dinger equation)'
#################################################################################################################

#################################################################################################################
'Importing packages'
#################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import constants
from scipy.linalg import expm
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import os


#################################################################################################################
'loading parameter file'
#################################################################################################################


params = {}

with open('Parameters exercise_4.txt', 'r') as file:
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
n = int(params.get('n'))
t = float(params.get('T'))*(femto/constants.value('atomic unit of time'))
x = float(params.get('x'))*(Ang/constants.value('atomic unit of length'))

nx = 2**(int(params.get('number_of_space_steps')))
nt = int(params.get('number_of_steps_t'))

t_points = np.linspace(0, t, nt)
x_points = np.linspace(-x, x, nx)

dx = float((x+x)/nx)
dt = (t_points[1]-t_points[0])

m = float(params.get('m')) 
                                                                                     ##mass 
C = float(params.get('C'))/((constants.value('atomic unit of force')/constants.value('atomic unit of length')))   ##spring_constant
omega = np.sqrt(C/m)
x0 = np.sqrt((2*omega*(n+1/2)/C))#float(params.get('x0'))*(Ang/constants.value('atomic unit of length')) 
                                    
                                                                                            ##angular frequency


################################################################################################################
'defining function for initial wavefunction and normalise'
################################################################################################################


'''Gaussian_wavefunction_generator_function'''


occupation = np.zeros((int(nt),2), dtype=np.complex128)




prefactor = (m * omega / np.pi)**0.25
exponentL = -((m * omega) * ((x_points + x0)**2)) / 2
psiL = prefactor * np.exp(exponentL)
normalization_constant = np.sqrt(np.trapz((abs(psiL))**2, dx=1))
psiL = psiL / normalization_constant


exponentR =  -((m * omega) * ((x_points - x0)**2)) / 2
psiR = prefactor * np.exp(exponentR)
normalization_constant = np.sqrt(np.trapz((abs(psiR))**2, dx=1))
psiR = psiR / normalization_constant

psi = psiL + psiR

# Normalize the wavefunction
normalization_constant = np.sqrt(np.trapz((abs(psi))**2, dx=dx))
normalized_psi = psi / normalization_constant

psi0 = psiL


c = np.zeros((int(nt),2), dtype=np.complex128)
################################################################################################################
'Potential function'
################################################################################################################


def v(x, C, x0):
    return (C / 2) * (np.abs(x) - x0)**2


################################################################################################################
'define Vo, K, Vo matrices'
################################################################################################################

####half_potential_operator
V = v(x_points, C, x0)
Vo = (-1j*V*dt/2)
Vop = np.exp(Vo)

####full kinetic operator
res = nx
dk = np.pi / x
K = np.concatenate((np.arange(0, res / 2),
                    np.arange(-res / 2, 0))) *dk

Ko = (-1j*((K**2)/(2*m))*dt)
Kop = np.exp(Ko)

def split_operator_propagation(wavefunction, Vop, Kop):
    P_basis = fft(Vop*wavefunction)
    x_basis = ifft(Kop*P_basis)
    wavefunction_at_dt = Vop*x_basis
    return wavefunction_at_dt


wavefunction_values = []
jj = 0
E0 = omega/2
eps = omega*n
for i in range(nt):
    
    wavefunction_step = split_operator_propagation(psi0 , Vop,  Kop)
    wavefunction_values.append(wavefunction_step)
    c[jj,0] = np.abs(np.real((1/np.sqrt(2))*np.exp(-1j*i*dt*(E0-eps))))**2
    c[jj,1] = np.abs(np.imag((1/np.sqrt(2))*np.exp(1J*i*dt*(E0+eps))))**2
    jj += 1
    initial_wavefunction = wavefunction_step

################################################################################################################
'Save frames as PNG images'
################################################################################################################
x_fector = (Ang/constants.value('atomic unit of length'))
V = (v(x_points, C, x0)*J)/e




for frame, psi in enumerate(wavefunction_values):
    fig, ax1= plt.subplots()
    ax1.plot(x_points/ x_fector, c[frame,0]*np.abs(psiL)**2)
    ax1.plot(x_points/ x_fector, c[frame,1]*np.abs(psiR)**2)
    ax1.set_xlabel('Position (10^(-10)m)')
    ax1.set_ylabel('Probability (|Î¨(x,t)|^2)(normalised)')
    plt.xlim(-x/x_fector, x/x_fector)
    plt.ylim(0, 0.05)
    
    
    
    ax2 = ax1.twinx()
    ax2.plot(x_points/ x_fector, V)
    #plt.ylim(0, 150)
    plt.grid(True)
    plt.title(f'Time Evolution - {round(((t/len(t_points)/41.34137333518211)*frame), 5)}fs')
   
    
    # Create a folder to save the figures
    output_folder = 'C:/Users/akbar/Downloads/pictures'
    os.makedirs(output_folder, exist_ok=True)
    # Save the plot in the specified folder
    output_filename = os.path.join(output_folder, f'frame_{frame:03d}.png')
    plt.savefig(output_filename)
    plt.close()
    














