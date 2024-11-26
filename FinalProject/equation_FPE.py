import numpy as np
from matrix import *
from simulation import *
import csv

# Heat diffusion setup
def heat_diffusion():
    x1_dt = np.array([[0, 0, 0]])
    x2_dt = np.array([[0, 0, 0]])
    D = [0.5, 0.5]
    dt = 0.01
    T_end = 25
    M1, M2 = 10, 10
    N1, N2 = 200, 200
    mu = [0, 0]
    sigma = (1 / 5) * np.eye(2)

    return M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma

# Linear oscillator setup
def linear_oscillator():
    x1_dt = np.array([[1, 0, 1]])
    x2_dt = np.array([[-0.2, 0, 1], [-1, 1, 0]])
    D = [0, 0.2]
    dt = 0.01
    T_end = 25
    M1, M2 = 10, 10
    N1, N2 = 200, 200
    mu = [5, 5]
    sigma = (1 / 9) * np.eye(2)

    return M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma

# Bimodal oscillator setup
def bimodal_oscillator():
    x1_dt = np.array([[1, 0, 1]])
    x2_dt = np.array([[1, 1, 0], [-0.4, 0, 1], [-0.1, 3, 0]])
    D = [0, 0.4]
    dt = 0.0075
    T_end = 15
    M1, M2 = 10, 15
    N1, N2 = 300, 300
    mu = [0, 10]
    sigma = (1 / 2) * np.eye(2)

    return M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma

# Van der Pol oscillator setup
def van_der_pol_oscillator():
    x1_dt = np.array([[1, 0, 1]])
    x2_dt = np.array([[-0.1, 2, 1], [0.1, 0, 1], [-1, 1, 0]])
    D = [0, 0.5]
    dt = 0.01
    T_end = 50
    M1, M2 = 10, 10
    N1, N2 = 200, 200
    mu = [4, 4]
    sigma = (1 / 2) * np.eye(2)

    return M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma

# Choose which setup to use by uncommenting one of the lines below:
M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma = heat_diffusion()
# M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma = linear_oscillator()
# M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma = bimodal_oscillator()
# M1, N1, M2, N2, x1_dt, x2_dt, D, dt, T_end, mu, sigma = van_der_pol_oscillator()

# Constructing matrices
D1, D2, D11, D22, x1, x2 = make_matrix(M1, N1, M2, N2, x1_dt, x2_dt, D)

# Solving the advection-diffusion problem
t, p = solve_advDif(D1, D2, D11, D22, x1, x2, dt, T_end, mu, sigma)

# Example usage for create_simulation
print(p[:, :, -1])

# Write the solution to a CSV file
with open('advection_diffusion_solution.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'X1', 'X2', 'Value'])
    for i in range(len(t)):
        for j in range(len(x1)):
            for k in range(len(x2)):
                writer.writerow([t[i], x1[j], x2[k], p[j, k, i]])
# create_simulation(x1, x2, t, p, skip_frame=8, from_top=True, writeVids=False, equal_axis=True)

# Explanation:
# This script sets up different scenarios for advection-diffusion simulations.
# Each section defines a different problem, e.g., "heat diffusion" or "linear oscillator," 
# with varying parameters such as advection coefficients (x1_dt, x2_dt), diffusion coefficients (D), 
# time step (dt), and total simulation time (T_end).
# 
# Once the problem is set, the matrices required for the solution are generated using `make_matrix()`.
# Finally, `solve_advDif()` is used to solve the problem over time.
#
# Note:
# You can uncomment any of the function calls to simulate a specific scenario.

