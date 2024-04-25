import numpy as np
import time

from scipy import linalg as LA
from scipy import optimize as Opt
from scipy import sparse as sp
import sys

from CGL_parameters import *
from integrators import *
from diff_mat import *
from matplotlib import pyplot as plt

# Parameters
x0 = -50                      # beginning of spatial domain
x1 = 50                       # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 20                       # Total simulation time
Tmax = 5

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
Nt = int(T / dt)                # number of timesteps
xvec = np.linspace(x0, x1, Nx)     # spatial grid
tvec = np.linspace(0, T, Nt)       # temporal grid

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

Ldir = np.matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)

data = np.load('Dynamics_matrix.npy')