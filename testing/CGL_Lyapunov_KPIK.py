import numpy as np
import sys
import time

from core.utils import pvec, pmat

from scipy import linalg
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.lyapunov import kpik

plt.close("all")

# Parameters
x0 = -10                      # beginning of spatial domain
x1 = 10                       # end of spatial domain
dx = 0.1                      # Spatial discretisation

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nxc = int(L / dx)                # number of spatial dof
xvec = np.linspace(x0, x1, Nxc)     # spatial grid

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.1
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__ = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

A = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

Ar = np.real(A)
Ai = np.imag(A)

# make real
A = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

# rhs
m = 1
B = np.zeros((Nx,m))
B[-1] = 400

tolY = 1e-12
tol  = 1e-10
k_max = 100

Z, err2, k_eff, etime = kpik(A,B,k_max,tol,tolY)

nstep = err2.shape[0]

print(' its |        comp.res. | space dim. |       CPU Time')
print(f'{nstep:4d} |   {err2[-1]:14.12f} |       {k_eff:4d} | {etime:14.12f}')
   
                     
