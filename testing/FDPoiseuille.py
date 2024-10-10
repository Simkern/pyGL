import numpy as np
import time
import sys

from scipy import linalg as LA
from matplotlib import pyplot as plt

import scipy.sparse as sp

# --> Differential operators.
def D2_1D(nx, dx, BC='dir'):
    
    # --> Diagonals of the finite-difference approximation.
    d, u = nx*[-2/dx**2], (nx-1)*[1/dx**2]

    # --> Laplacian matrix.
    L = np.diag(d) + np.diag(u, k=1) + np.diag(u, k=-1)

    # --> Periodic boundary conditions?
    if BC == 'W':
        pass
    elif BC == 'P':
        L[0, -1] = 1/dx**2
        L[-1, 0] = 1/dx**2
    else:
        print('Undefined BC.')

    return L

def D_1D(nx, dx, BC='dir'):
    
    # --> Diagonals of the finite-difference approximation.
    u = (nx-1)*[1/(2*dx)]
    l = (nx-1)*[-1/(2*dx)]

    # --> Differentiation matrix.
    D = np.diag(u, k=1) + np.diag(l, k=-1)

    # --> Periodic boundary conditions?
    if BC == 'W':
        pass
    elif BC == 'P':
        D[0, -1] = -1/(2*dx)
        D[-1, 0] = 1/(2*dx)
        
    return D

def F(U, D2y, Re, dpdx):
    
    fu = D2y/Re @ U + dpdx*np.ones(u.shape)
    
    return fu

def Ff(U, omega, D2y, Re, dpdx):
    
    fu = D2y/Re @ U - dpdx*np.ones(u.shape) - 1j*omega*U
    
    return fu

def dF(U, D2y, Re):
    
    return D2y/Re

def dFf(U, omega, D2y, Re):
    
     return D2y/Re - np.diag(1j*omega*np.ones(u.shape))

if __name__ == "__main__":
    # --> Parameters.
    ny = 31
    Ly = 2.0
    dy = Ly/(ny+1)
    Re = 100.0
    G  = 2/Re
    omega = 0.24

    # --> Differential operators.
    Dy = D_1D(ny, dy, BC='W')
    D2y = D2_1D(ny, dy, BC='W')

    # --> Mesh.
    y = np.linspace(-Ly/2, Ly/2, ny+2)[1:-1]
    
    # --> Baseflow velocity profile.
    u_BF = 1 - y**2
    du_BF = -2*y

    # --> Orr-Sommerfeld Squire operators.
    OS = D2y/Re
    
    #u = np.ones(y.shape)
    u = np.random.rand(ny)
    plt.plot(y, u)

    fu = F(u, D2y, Re, G)
    
    print(f'norm: {np.linalg.norm(fu)}')
        
    J = dF(fu, D2y, Re)

    du = LA.solve(J, -fu)

    u = u + du
    plt.plot(y, u)

    fu = F(u, D2y, Re, G)
    print(f'norm: {np.linalg.norm(fu)}')
    
    
    u = np.random.rand(ny)
    plt.plot(y, u)
    ffu = Ff(u, omega, D2y, Re, G)
    print(f'norm: {np.linalg.norm(fu)}')
        
    Jf = dFf(ffu, omega, D2y, Re)

    du = LA.solve(Jf, -ffu)

    u = u + du
    plt.plot(y, u)

    ffu = Ff(u, omega, D2y, Re, G)
    print(f'norm: {np.linalg.norm(fu)}')

#fu = F(u, DM1f, DM2c, nu, 1, 0)
#print(f'norm: {np.linalg.norm(fu)}')