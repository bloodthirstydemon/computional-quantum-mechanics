# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:41:00 2023

@author: akbar
"""

import numpy as np
import matplotlib.pyplot as plt

'''firts step is to generate wigner distribution 
for a pure quantum state without any potential
for example: gaussian wave packet'''

def wavefunction(x, x0, sigma, p0, h_bar=1.0):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * x / h_bar)


def wigner_distribution(psi, x, p, h_bar=1.0):
    W = np.zeros((len(x), len(p)), dtype=complex)
    dx = x[1] - x[0]

    for i in range(len(x)):
        for j in range(len(p)):
            W[i, j] = (1.0 / (np.pi * h_bar)) * np.trapz(
                       np.conj(psi) * np.roll(psi, i) * np.exp(-1j * p[j] * x / h_bar), x)

    return W


x_min, x_max = -10, 10
p_min, p_max = -5, 5
x_points = 200
p_points = 200
x = np.linspace(x_min, x_max, x_points)
p = np.linspace(p_min, p_max, p_points)

# Define parameters for the Gaussian wave packet
x0 = 0.0       # Initial position
sigma = 1.0    # width of the gaussian distribution
p0 = 1       # Initial momentum
h_bar = 1.0    # Reduced Planck's constant

# Generate Gaussian wave packet
psi_gaussian = wavefunction(x, x0, sigma, p0, h_bar)

# Calculate the Wigner distribution for the Gaussian wave packet
W = wigner_distribution(psi_gaussian, x, p, h_bar)

# Plot the Gaussian wave packet and its Wigner distribution

# Plot the Wigner distribution

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, p, np.abs(W), cmap='viridis', rstride=5, cstride=5, alpha=0.7, antialiased=True)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Momentum (p)')
ax.set_zlabel('Absolute magnitude')
ax.set_title('Wigner Distribution')
plt.show()