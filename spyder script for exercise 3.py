#################################################################################################################
'Homework_4(Split-Operator method to solve time dependent schrödinger equation)'
#################################################################################################################

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

t = float(params.get('T'))*(femto/constants.value('atomic unit of time'))
x = float(params.get('x'))*(Ang/constants.value('atomic unit of length'))

nx = 2**(int(params.get('number_of_space_steps')))
nt = int(params.get('number_of_steps_t'))

t_points = np.linspace(0, t, nt)
x_points = np.linspace(-x, x, nx)

dx = float((x+x)/nx)
dt = (t_points[1]-t_points[0])

m = float(params.get('m'))                                                                                      ##mass 
C = float(params.get('C'))/((constants.value('atomic unit of force')/constants.value('atomic unit of length')))   ##spring_constant
x0 = float(params.get('x0'))*(Ang/constants.value('atomic unit of length')) 
s0 = float(params.get('s0'))*(Ang/constants.value('atomic unit of length'))                                    ##centre_of_wavepacket 
omega = np.sqrt(C/m)                                                                                            ##angular frequency


################################################################################################################
'Potential function'
################################################################################################################


def V(C, x, x0):
    return (C / 2) * (np.abs(x) - x0)**2


################################################################################################################
'defining function for initial wavefunction and normalise'
################################################################################################################


'''Gaussian_wavefunction_generator_function'''


def wavefunction(x, m, omega, s0):
    prefactor = (m * omega / np.pi)**0.25
    exponent = -((m * omega) * ((x - s0)**2)) / 2
    return prefactor * np.exp(exponent)


'''function_that_can_normalise_any_given_wavefunction_using_trapazoidal_rule'''


def normalize_wavefunction(psi):
    # Calculate the normalization constant
    #normalization_constant = np.sqrt(scipy.integrate.simps(abs(psi) ** 2), dx)
    
    
    normalization_constant = np.sqrt(np.trapz((abs(psi))**2, dx=dx))

    '''Here it is imporatnt to note that dx(integration parameter) is taken as 1,
    because if we want to take integration(psi*dx) then dx should be physically, gap between two descrete basis vectors
    which would be to small and then integration would be to small 
    and normalised wavefunction would explode giving us probabilities larger than 1.'''

    # Normalize the wavefunction
    normalized_psi = psi / normalization_constant

    return normalized_psi



################################################################################################################
'split_operator propogation'
################################################################################################################


def split_operator_propagation(wavefunction, Vop, Kop):
    p_basis = np.fft.fftshift(np.fft.fft(Vop @ wavefunction))/(2*np.pi)
    x_basis = np.fft.ifft(Kop @ p_basis)/(2*np.pi)
    wavefunction_at_dt = normalize_wavefunction(Vop @ x_basis)
    
    return wavefunction_at_dt


################################################################################################################
'define Vo, K, Vo matrices'
################################################################################################################




res = len(x_points)
dk = np.pi / x
k = np.fft.fftshift(np.fft.fftfreq(len(x_points), dx/2*np.pi ))#, 

#####define operators
Ko = np.exp(-1j*((k**2)/(2*m))*dt)
Kop = np.diag(Ko)
Vo = np.exp(-1j*V(C, x_points, x0)*dt/2)
Vop = np.diag(Vo)


################################################################################################################
'implementation'
################################################################################################################


wavefunction_values = []
initial_wavefunction = normalize_wavefunction(wavefunction(x_points, m, omega, s0))
#%%
for i in range(nt):
    
    wavefunction_step = split_operator_propagation(initial_wavefunction , Vop,  Kop)
    wavefunction_values.append(wavefunction_step)
    initial_wavefunction = wavefunction_step


################################################################################################################
'Save frames as PNG images'
################################################################################################################
x_fector = (Ang/constants.value('atomic unit of length'))
V = (V(C, x_points, x0)*J)/e




for frame, psi in enumerate(wavefunction_values):
    fig, ax1= plt.subplots()
    ax1.plot(x_points/ x_fector, np.abs(psi)**2)
    ax1.set_xlabel('Position (10^(-10)m)')
    ax1.set_ylabel('Probability (|Ψ(x,t)|^2)(normalised)')
    plt.xlim(-x/x_fector, x/x_fector)
    plt.ylim(0, max(np.abs(psi)**2))
    
    ax2 = ax1.twinx()
    ax2.plot(x_points, V)
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