#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
from scipy import linalg as LA

from git.core.CGL_parameters import *
from git.core.diff_mat import *
from git.solvers.arnoldi import arn, arn_inv
from matplotlib import pyplot as plt

def compute_shifts(na, nb, A, Nx, npick):
    
    # compute shifts
    b0 = np.ones((Nx,))
    # Compute the set of Ritz values R+ for A
    na = 60
    __, Ha = arn(A, b0, na)
    pA, __ = LA.eig(Ha[0:na, 0:na])
    pA0 = sorted(pA)
    # Compute the set of Ritz values R- for A^-1
    nb = 60
    __, Hb = arn_inv(A, b0, nb)
    Dbtmp, __ = LA.eig(Hb[0:nb, 0:nb])
    pAinv0 = np.array([1/r for r in sorted(Dbtmp)])

    pA = np.array([], dtype=complex)
    for i, p in enumerate(pA0):
        if np.isreal(p):
            pA = np.append(pA, p)
        else:
            pA = np.append(pA, p)
            pA = np.append(pA, p.conj())
    pAinv = np.array([], dtype=complex)
    for i, p in enumerate(pAinv0):
        if np.isreal(p):
            pAinv = np.append(pAinv, p)
        else:
            pAinv = np.append(pAinv, p)
            pAinv = np.append(pAinv, p.conj())

    idx = np.argsort(-np.real(pA))
    pA = pA[idx]
    idx = np.argsort(-np.real(pAinv))
    pAinv = pAinv[idx]

    pA_v = np.empty((npick,), dtype='complex')

    eps = 1e-12
    ip = 0
    i = 0
    pAnreal = 0
    pAncmplx = 0
    while ip < npick and i < pA.size:
        p = pA[i]
        if np.abs(np.imag(p)) < eps:
            p = np.real(p)
        if np.real(p) > 0:
            p = -p
        if np.isreal(p):
            if not np.isclose(pA_v, p, atol=eps).any():
                pA_v[ip] = p
                ip += 1
                pAnreal += 1
        else:
            if not np.isclose(pA_v, p, atol=eps).any() and not np.isclose(pA_v, p.conj(), atol=eps).any():
                pA_v[ip] = p
                ip += 1
                pAncmplx += 1

        i += 1
    if i == pA.size:
        print('Not enough shifts to fill pA_v array')
        sys.exit()
    print('pA_v:')
    print(f'  Number of available shifts:     {pA.size:3d}')
    # pvec(pA)
    print(f'  Shifts with unique modulus:     {npick:3d},  ', end='')
    print(f'  Total:  {pAnreal + 2*pAncmplx:3d},  ', end='')
    print(f'  ({pAnreal} real / {pAncmplx} complex conjugate)')
    # pvec(pA_v)

    pAinv_v = np.empty((npick,), dtype='complex')
    ip = 0
    i = 0
    pAinvnreal = 0
    pAinvncmplx = 0
    while ip < npick and i < pAinv.size:
        p = pAinv[i]
        if np.abs(np.imag(p)) < eps:
            p = np.real(p)
        if np.real(p) > 0:
            p = -p
        if np.isreal(p):
            if not np.isclose(pAinv_v, p, atol=eps).any():
                pAinv_v[ip] = p
                ip += 1
                pAinvnreal += 1
        else:
            if not np.isclose(pAinv_v, p, atol=eps).any() and not np.isclose(pAinv_v, p.conj(), atol=eps).any():
                pAinv_v[ip] = p
                ip += 1
                pAinvncmplx += 1

        i += 1
    if i == pAinv.size:
        print('Not enough shifts to fill PAinv_v array')
        sys.exit()

    l = pAinv_v.size
    print('pAinv_v:')
    print(f'  Number of available shifts:     {pAinv.size:3d}')
    print(f'  Shifts with unique modulus:     {npick:3d},  ', end='')
    print(f'  Total:  {pAnreal + 2*pAncmplx:3d},  ', end='')
    print(f'  ({pAnreal} real / {pAncmplx} complex conjugate)')

    return pA, pAinv, pA_v, pAinv_v

# Parameters
x0 = -30                       # beginning of spatial domain
x1 = 30                         # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 20                         # Total simulation time

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
Nt = int(T / dt)                # number of timesteps
x = np.linspace(x0, x1, Nx)     # spatial grid
t = np.linspace(0, T, Nt)       # temporal grid

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.1
cd  = -1
mu0 = 0.38
mu2 = -0.01

# Get derived parameters
mu,nu,gamma,Umax,mu_t = CGL2(x,mu0,cu,cd,U,mu2,False)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(x)

# Form system matrix (direct)
L  = np.asarray((np.diag(mu) - nu*DM1f + gamma*DM2c))
# Compute spectrum
D,__  = LA.eig(L)

idx = np.argsort(-np.real(D))
D   = D[idx]

b0 = np.ones((L.shape[0],))
# Compute the set of Ritz values R+ for A
ka = 50
__, Ha = arn(L,b0,ka)
Da,__  = LA.eig(Ha[0:ka,0:ka])

# Compute the set of Ritz values R- for A^-1
kb = 20
__, Hb   = arn_inv(L,b0,kb)
Dbtmp,__ = LA.eig(Hb[0:kb,0:kb])
Db = [ 1/r for r in Dbtmp ]

fig = plt.figure(1)
ax = plt.plot(np.real(Da),np.imag(Da),'ro', mfc='none',  label='Ritz values R+')
plt.plot(np.real(Db),np.imag(Db),'bo', mfc='none',  label='Ritz values 1/R-')
plt.plot(np.real(D),np.imag(D),'k+', label = 'Eigenvalues A')
plt.axis('square')
plt.legend()
plt.show()



