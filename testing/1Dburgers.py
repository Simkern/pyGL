import numpy as np
import time
import sys

from scipy import linalg as LA
from matplotlib import pyplot as plt

sys.path.append('..')

from core.diff_mat import FDmat

def F(U, DM1f, DM2c, nu, u0, uL):
    f = np.diag(U) * DM1f @ U - nu*DM2c @ U
    #f[0] = u0
    #f[-1] = uL
    return f

def J(U, DM1f, DM2c, nu):
    A = 2*np.diag(U) @ DM1f - nu*DM2c
    return A

plt.close('all')

# Parameters
x0 = -1                      # beginning of spatial domain
x1 = 1                       # end of spatial domain
dx = 0.01                      # Spatial discretisation
nu = 0.05

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
x = np.linspace(x0, x1, Nx)     # spatial grid

# Generate differentiation matrices
D2c = sp.diags([1, -2, 1],[-1, 0, 1], shape = (Nx,Nx))

u = -np.tanh(1/(2*nu)*x)# + np.exp(-x**2*10)/10

plt.plot(x, u)


fu = F(u, DM1f, DM2c, nu, 1, -1)
plt.plot(x, fu)
print(f'norm: {np.linalg.norm(fu)}')
sys.exit()
Ju = J(fu, DM1f, DM2c, nu)

du = LA.solve(Ju, fu)

u = u - du

fu = F(u, DM1f, DM2c, nu)
print(f'norm: {np.linalg.norm(fu)}')

Ju = J(fu, DM1f, DM2c, nu)

du = LA.solve(Ju, fu)

u = u - du

fu = F(u, DM1f, DM2c, nu)
print(f'norm: {np.linalg.norm(fu)}')

Ju = J(fu, DM1f, DM2c, nu)

du = LA.solve(Ju, fu)

u = u - du

fu = F(u, DM1f, DM2c, nu)
print(f'norm: {np.linalg.norm(fu)}')

Ju = J(fu, DM1f, DM2c, nu)

du = LA.solve(Ju, fu)

u = u - du

fu = F(u, DM1f, DM2c, nu)
print(f'norm: {np.linalg.norm(fu)}')

Ju = J(fu, DM1f, DM2c, nu)

du = LA.solve(Ju, fu)

u = u - du

#fu = F(u, DM1f, DM2c, nu, 1, 0)
#print(f'norm: {np.linalg.norm(fu)}')