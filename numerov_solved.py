


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
h = (x_max - x_min) / (num_points)
x = np.linspace(x_min, x_max, int(num_points))
matching_point = int(len(x)/2)+50

################################################################################################################
'Potential function'
################################################################################################################

def potential(x):
    return 0.5 * C * x**2 - 0.5 * D * x**3 + 0.5 * E * x**4


################################################################################################################
'Numerov method for solving Schrödinger equation'
################################################################################################################


def numerov_forward(x, e, h):
    psi = np.zeros_like(x, dtype=np.float64)  # Use double precision
    V = -2 * m * (e - potential(x))

    # Initial values for Numerov method
    psi[0] = 0
    psi[1] = 0.0000001

    for i in range(2, matching_point+3):

      psi[i] =  (2 * psi[i-1] - psi[i - 2] + (h**2/(12)) * (10 * V[i-1] * psi[i - 1] + V[i-2] * psi[i - 2])) / (1 - (h**2/(12)) * V[i])


    return psi


def numerov_backward(x, e, h):
    psi = np.zeros_like(x, dtype=np.float64)  # Use double precision
    V = -2 * m * (e - potential(x))

    # Initial values for Numerov method
    psi[-1] = 0
    psi[-2] = 0.0000001

    for i in range(3, matching_point-100+2):

      psi[-i] =  (2 * psi[-(i - 1)] - psi[-(i - 2)] + (h**2/(12)) * (10 * V[-(i-1)] * psi[-(i - 1)] + V[-(i-2)] * psi[-(i - 2)])) / (1 - (h**2/(12)) * V[-i])


    return psi 


################################################################################################################
'looping over energies to find eigen energy'
################################################################################################################

  
e = np.zeros((5, 2000))

# Fill the matrix with values
for i in range(5):
    e[i, :] = np.linspace(i ,i+1, 2000)


# =============================================================================


fig, axs = plt.subplots(5, 1, figsize=(10, 12))

for i in range(5):
    for j in range(2000):
        energy = e[i, j]

        psi_forward = numerov_forward(x, energy, h)
        psi_backward = numerov_backward(x, energy, h)
 
     
        if i %2 == 0:
            ratio = (psi_forward[matching_point]) / (psi_backward[-matching_point+100+1])
            psif = (- psi_forward[matching_point+2] + 8 * psi_forward[matching_point+1] - 8 * psi_forward[
                    matching_point-1] + 8 * psi_forward[matching_point- 2]) / 12 * h             
            psib = (- psi_backward[-matching_point+100-1] + 8 * psi_backward[-matching_point+100] - 8 * psi_backward[
                    -matching_point+100+ 2] + 8 * psi_backward[-matching_point+100+ 3]) / 12 * h
        else:
                 
            ratio = np.abs(psi_forward[matching_point]) / np.abs(psi_backward[-matching_point+100-1])
            psif = (- psi_forward[matching_point] + 8 * psi_forward[matching_point-1] - 8 * psi_forward[
                    matching_point-3] + 8 * psi_forward[matching_point- 4]) / 12 * h
            psib = (- psi_backward[-matching_point+100-1] + 8 * psi_backward[-matching_point+100] - 8 * psi_backward[
                    -matching_point+100+ 2] + 8 * psi_backward[-matching_point+100+ 3]) / 12 * h
 
                
        if 0.9367280566893119 <= np.abs(psif / psib) <= 1.4 and 0.94 <= ratio <= 1.0001:
            axs[i].plot(x, numerov_forward(x, energy, h), label=f"{i}_eigen_state,\n Numerov Forward, Energy: {energy}")
            axs[i].plot(x, ((-1) ** i) * numerov_backward(x, energy, h), label=f"Numerov Backward, Energy: {energy}")
            axs[i].legend()
            
            print(f'{i}_eigen_state')
            print(f'{i}_{energy}')
            print(f'{i}_')
            print(f'{i}_{psif/psib}')
            print(f'{i}_{ratio}')
            
            break  

plt.tight_layout()
plt.show()




#%%    
i = 1
energy = 2.5
psi_forward = numerov_forward(x, energy, h)
psi_backward = numerov_backward(x, energy, h)
ratio = (psi_forward[matching_point]) / (psi_backward[-matching_point+100+1])
psif = (- psi_forward[matching_point+2] + 8 * psi_forward[matching_point+1] - 8 * psi_forward[
        matching_point-1] + 8 * psi_forward[matching_point- 2]) / 12 * h             
psib = (- psi_backward[-matching_point+100-1] + 8 * psi_backward[-matching_point+100] - 8 * psi_backward[
        -matching_point+100+ 2] + 8 * psi_backward[-matching_point+100+ 3]) / 12 * h
plt.plot(x, numerov_forward(x, energy, h))
plt.plot(x, numerov_backward(x, energy, h))


    
# =============================================================================
# 
# 
# 
#     
#         if i %2 == 0:
#             ratio = (psi_forward[matching_point]) / (psi_backward[-matching_point+1])
#             psif = (- psi_forward[matching_point+2] + 8 * psi_forward[matching_point+1] - 8 * psi_forward[
#                     matching_point-1] + 8 * psi_forward[matching_point- 2]) / 12 * h
#             psib = (- psi_backward[-matching_point-1] + 8 * psi_backward[-matching_point] - 8 * psi_backward[
#                     -matching_point+ 2] + 8 * psi_backward[-matching_point+ 3]) / 12 * h
#         else:
#             if -2.98e11>psi_forward[matching_point]<2.98e11:
#                 
#                 ratio = (psi_forward[matching_point]) / (psi_backward[-matching_point-1])
#                 psif = (- psi_forward[matching_point] + 8 * psi_forward[matching_point-1] - 8 * psi_forward[
#                         matching_point-3] + 8 * psi_forward[matching_point- 4]) / 12 * h
#                 psib = (- psi_backward[-matching_point-1] + 8 * psi_backward[-matching_point] - 8 * psi_backward[
#                         -matching_point+ 2] + 8 * psi_backward[-matching_point+ 3]) / 12 * h
# 
#             else:
#                 ratio = 5
#                 
# 
# 
# =============================================================================



# =============================================================================
#     
#     
#     
# for i in range(5):
#   for j in range(2000):
#     energy = e[i,j]
# 
#     psi_forward = numerov_forward(x, energy, h)
#     psi_backward = numerov_backward(x, energy, h)
#     ratio = np.abs(psi_forward[-int((len(x)/2)+1)]) / np.abs(psi_backward[-int((len(x)/2))])
#     psif = (- psi_forward[int(len(x)/2)] + 8*psi_forward[int(len(x)/2)-1] - 8*psi_forward[int(len(x)/2)-2] + 8*psi_forward[int(len(x)/2)-3])/12*h
#     psib = (- psi_backward[-int(len(x)/2)] + 8*psi_backward[-int(len(x)/2)+1] - 8*psi_backward[-int(len(x)/2)+2] + 8*psi_backward[-int(len(x)/2)+3])/12*h
#     
#     if 0.9999 <= psif / psib <= 1.00001 and 0.9999 <= ratio <= 1.00001:
#             plt.plot(x, numerov_forward(x, energy, h), label="Numerov Forward")
#             plt.plot(x, ((-1) ** i) * numerov_backward(x, energy, h), label="Numerov Backward")
#             print(f'{i}_{energy}')
#             print(f'{i}_{psif/psib}')
#             print(f'{i}_{ratio}')
#             plt.legend()
#             plt.show()
#             break
# 
# =============================================================================
