import numpy as np
import scipy.sparse as sp

def FDmat(x):
    Nx = len(x)
    dx = x[2]-x[1]
    
    assert np.max(np.diff(x)) - np.min(np.diff(x)) < 1e-12, \
        f'Grid is not equidistant. Do not use this FD method.'

    # stencils
    D1f,D1b = DM1(Nx)
    D2c     = DM2(Nx)
    # normalize
    D1f = D1f/(2*dx)
    D1b = D1b/(2*dx)
    D2c = D2c/dx**2

    return D1f,D1b,D2c

# first derivative (second order upwind)
def DM1(Nx):
    D1f = sp.diags([-3, 4, -1],[0, 1, 2], shape = (Nx,Nx))
    #D1f = sp.diags([-1, 1],[0, 1], shape = (Nx,Nx))
    D1f = sp.lil_matrix(D1f)
    D1b = sp.diags([1, -4, 3],[-2, -1, 0], shape = (Nx,Nx))
    #D1b = sp.diags([-1, 1],[-1, 0], shape = (Nx,Nx))
    D1b = sp.lil_matrix(D1b)
    return D1f,D1b

# second derivative (second order)
def DM2(Nx):
    D2c = sp.diags([1, -2, 1],[-1, 0, 1], shape = (Nx,Nx))
    D2c = sp.lil_matrix(D2c)
    return D2c