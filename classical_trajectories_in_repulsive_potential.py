#################################################################################################################
'Project: Coupled Electron Nuclear Dynamics'
#################################################################################################################

#################################################################################################################
'Importing packages'
#################################################################################################################


import numpy as np
from scipy import constants
import scipy
from scipy.linalg import solve
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
import seaborn as sns


#################################################################################################################
'loading parameter file'
#################################################################################################################


params = {}

with open('project.txt', 'r') as file:
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
q = np.linspace(-x, x, nx)


dx = float((x+x)/nx)
dt = (t_points[1]-t_points[0])

p = np.linspace(-np.pi/dx, np.pi/dx, nx)

m = float(params.get('m'))                                                                                      ##mass 
C = float(params.get('C'))/(constants.value('atomic unit of force')/constants.value('atomic unit of length'))   ##spring_constant
x0 = float(params.get('x0'))*(Ang/constants.value('atomic unit of length')) 
s0 = float(params.get('s0'))*(Ang/constants.value('atomic unit of length'))                                    ##centre_of_wavepacket 
omega = np.sqrt(C/m)                                                                                            ##angular frequency

#%%Function_definitions
################################################################################################################
'defining function for initial wavefunction and normalise'
################################################################################################################


'''Gaussian_wavefunction_generator_function'''


def wavefunction(q, m, omega, s0):
    prefactor = (m * omega / np.pi)**0.25
    exponent = -((m * omega) * (q - s0)**2) / 2
    psi = prefactor * np.exp(exponent)
    normalization_constant = np.sqrt(scipy.integrate.simps(abs(psi) ** 2))
    psi = psi / normalization_constant
    return psi


################################################################################################################
'Wigner_function implementation'
################################################################################################################


def wigner_distribution(q, p):
    N = 1                                                                                 # Number of dimensions
    s_values = np.linspace(-x, x, nx)
    integral = np.trapz(np.exp(1j * p * s_values)
                        * wavefunction((q - s_values / 2), m, omega, s0)
                        * wavefunction((q + s_values / 2), m, omega, s0), s_values)
    
    return ((1 / (2 * np.pi))**N) * (integral)


################################################################################################################
'calculate_total_energy'
################################################################################################################


def total_energy(q, p):
    return np.sum((0.5 * p**2 + 0.5 * q**2)*omega)


################################################################################################################
'orthant_sampling'
################################################################################################################



def orthant_sampling(N, num_samples):
    samples = []
    while len(samples) < num_samples:
        q_interval = (-0.2, 0.2)
        p_interval = (-7, 7)
        
        random_vector = (np.random.normal(0, 0.1),
                         np.random.normal(0, 3.5))


        q_sample = random_vector[0]
        p_sample = random_vector[1]

        samples.append((q_sample, p_sample))

    return np.array(samples)



# =============================================================================
# 
# def orthant_sampling(N, num_samples):
#     samples = []
#     while len(samples) < num_samples:
#         q_interval = (-1, 1)
#         p_interval = (-1, 1)
#         
#         
# 
#         # Generate random 2D vector
#         random_vector = (np.random.uniform(q_interval[0], q_interval[1]),
#                          np.random.uniform(p_interval[0], p_interval[1]))
#         magnitude = np.sqrt(random_vector[0]**2 + random_vector[1]**2)
# 
# 
#         # Step 2: Select a random point on the energy shell
#         q_sample = random_vector[0]/magnitude
#         p_sample = random_vector[1]/magnitude
# 
#         # Step 3: Evaluate Wigner distribution
#         wigner_value = wigner_distribution(q_sample, p_sample)
# 
#         # Step 4: von Neumann's rejection method
#         acceptance_ratio = wigner_value / wigner_distribution(0, 0)  # Assuming most probable value at (0, 0)
#         Ri = np.random.rand()
#         if acceptance_ratio <= Ri:
#             samples.append((q_sample, p_sample))
# 
#     return np.array(samples)
# 
# 
# =============================================================================

#%%Calculate_Wigner_distribution
################################################################################################################
'wigner distribution'
################################################################################################################

W = np.zeros((nx, nx), dtype=complex)
for i in range(nx):
    for j in range(nx):
        W[i,j] = wigner_distribution(q[i], p[j])

#%%      ploting wigner distribution  
################################################################################################################
'ploting wigner distribution'
################################################################################################################


plt.figure()
plt.contourf(q/(Ang/constants.value('atomic unit of length')), p, np.abs(W)**2, cmap='viridis', levels=nx)
plt.xlabel('Q (in A°)')
plt.ylabel('P (in atomic units)')
plt.title('Wigner Distribution')
plt.xlim(-.5, .5)
plt.ylim(-10,10)
plt.colorbar(label='W(q,p)')
plt.show()
 

#%%implementation of sampling
################################################################################################################
'implementation of sampling'
################################################################################################################


# Create a sample using orthant sampling
N = 1                                                                                     # Number of dimensions
num_samples = 300
samples = orthant_sampling(N, num_samples)

# Plot the sampled points on the energy shell

plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], marker='x')
plt.xlim(-max(samples[:, 0])-.5, max(samples[:, 0])+.5)
plt.title('Orthant Sampling on the Energy Shell')
plt.xlabel('q in atomic units')
plt.ylabel('p in atomic units')
plt.show()



#%%propagate_classical_trajectory
################################################################################################################
'propagate_classical_trajectory'
################################################################################################################
'''define potential that describes our system the most appropriately V = Vbond + Vbend + Vvan_der_vals + V_tortion + V_electro_static''' 

A = 1000                                                    #A has units of force_in_a.u.*atomic_unit_of_lenght**13

def disassociation_potential(q):
    r = q
    return A/r**12

def force(q):
    r = q
    return -12*A/r**13


################################################################################################################
'numerical integration (Eulers method) for disassociation_potential'
################################################################################################################


def propagate_classical_trajectory(q, p, dt, num_steps):
    trajectories = [(q, p)]

    for _ in range(num_steps):
        q = q + dt * (p/m)                               # Update position and momentum using Euler's method
        p = p - dt * force(q)                            # Update momentum using gradient of Coulomb potential

        trajectories.append((q, p))

    return np.array(trajectories)


################################################################################################################
'calculation_and_plotting'
################################################################################################################


# Propagate each sampled point classically
ensemble_trajectories = []

for i in range(num_samples):
    q_initial, p_initial = samples[i]  # Time step
    trajectory = propagate_classical_trajectory(q_initial, p_initial, dt, nt)
    ensemble_trajectories.append(trajectory)

# Plot the ensemble of classical trajectories
for i in range(num_samples):
    plt.plot(ensemble_trajectories[i][:, 0], ensemble_trajectories[i][:, 1])
    plt.yscale('log')
    plt.xlim(0,0.5)
    
    
#plt.plot(q, disassociation_potential(q), label='Coulomb Potential', color='red')

plt.figure()
plt.plot(q, disassociation_potential(q))
plt.yscale('log')
plt.title('Classical Trajectories for Sampled Points')
plt.xlabel('Position (q)')
plt.ylabel('Momentum (p)')
plt.show()
#%%
final_q_coordinate = []
final_p_coordinate = []
for i in range(num_samples):
    final_q_coordinate.append(np.asarray(ensemble_trajectories)[i,2,0])
    final_p_coordinate.append(np.asarray(ensemble_trajectories)[i,2,1])
sns.displot(final_q_coordinate, bins=50, kde=True)
sns.displot(final_p_coordinate, bins=50, kde=True)

