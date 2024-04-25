#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:32:31 2023

@author: skern
"""
import numpy as np
import time


from scipy import linalg as LA
from scipy import sparse as sp
from scipy import optimize as Opt
import sys

from CGL_parameters import *
from integrators import *
from diff_mat import *
from matplotlib import pyplot as plt

# Parameters
x0 = -30                      # beginning of spatial domain
x1 = 30                       # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 200                      # Total simulation time
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
mu0 = 0.41
mu2 = -0.01

mu_scal,__,__,__,__ = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

L = np.matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)

# Intialise fields
q = np.zeros((Nx,Nt), dtype=complex)
ql = np.zeros((Nx,Nt), dtype=complex)

D,X = np.linalg.eig(L)

Xinv = LA.inv(X)
tmax_global = Opt.minimize_scalar(lambda t: -LA.norm(X @ np.diag(np.exp(t*D)) @ Xinv, ord=2), bounds = [1, 50], method='bounded')
Phi_global = LA.expm(tmax_global.x*L)
U,S,Vh = LA.svd(Phi_global)
OIC_global = Vh[0,:].T
print(f'Time for maximum energy amplification: T = {tmax_global.x:.2f}')

q0 = OIC_global

q[:,0] = q0
ql[:,0] = q0

L = sp.lil_matrix(L)
etime = 0
for it in range(Nt-1):
    if it+1 % int(Nt/10) == 0:
        p = it/Nt*100
        end = time.time()
        etime = etime + (end-start)
        start = end
        print(f't = {tvec[it]:.1f}   ...   {p:.0f}%   ...   etime = {etime:.2f}')
    q[:,it+1] = CN_NL_advance(q[:,it],L,dt)
    ql[:,it+1] = CN_L_advance(ql[:,it],L,dt)

E = np.zeros((Nt,2))
for it in range(Nt):
    E[it,0] = np.real(np.inner(q[:,it].conj(),q[:,it]))
    E[it,1] = np.real(np.inner(ql[:,it].conj(),ql[:,it]))
    

Ntp = 2000
Nstep = int(Nt/Ntp)

tt,xx = np.meshgrid(tvec[1::Nstep],xvec)
plt.set_cmap('bwr')
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(xx,tt,np.real(q[:,1::Nstep]),cmap='bwr')

