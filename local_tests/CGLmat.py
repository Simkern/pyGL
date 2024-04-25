# -*- coding: utf-8 -*-
import numpy as np

from git.core.CGL_parameters import CGL, CGL2
from git.core.diff_mat import FDmat


def getCGLmatrices():
    
    # Parameters
    x0 = -5                      # beginning of spatial domain
    x1 = 5                      # end of spatial domain
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
    mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,False)
    x12 = np.sqrt(-2*mu_scal/mu2)

    # Generate differentiation matrices
    DM1f,DM1b,DM2c = FDmat(xvec)

    A = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

    Ar = np.real(A)
    Ai = np.imag(A)

    # make real
    A  = np.block([[Ar, -Ai],[Ai, Ar]])
    Nx = 2*Nxc

    # Laplacian
    Lc = np.asarray(DM2c.todense())
    Lr = np.real(Lc)
    Li = np.imag(Lc)
    L = np.block([[Lr, -Li], [Li, Lr]])
    
    return A,L,Lc,Nx,Nxc