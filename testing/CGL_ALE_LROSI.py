#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
from solvers.lyapunov import LR_OSI

def rn(X,Xref):
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

plt.close("all")


# Parameters
L  = 50
x0 = -L/2                      # beginning of spatial domain
x1 = L/2                       # end of spatial domain
Nxc = 128
xvec = np.linspace(x0, x1, Nxc+2)

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.2
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__   = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

A0 = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

# make real
Ar = np.real(A0)
Ai = np.imag(A0)
A = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

# random lhs of rank rk
rk = 5
B = np.random.random_sample((Nx,rk))

Q = B @ B.T

# random matrix of rank rk = 10
s0    = np.random.random_sample((rk,))
U0, _ = linalg.qr(np.random.random_sample((Nx, rk)),mode='economic')
S0    = np.diag(s0);
X0    = U0 @ S0 @ U0.T

nQ    = np.linalg.norm(Q)
nA    = np.linalg.norm(A)

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

# compare RK45 to LR_OSI
Tend = 0.01
tspan = (0,Tend)
tol = 1e-12
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=tol, rtol=tol)
Xrk = sol.y[:,-1].reshape(A.shape)
Urk,Srk,Vrkh = linalg.svd(Xrk)

rkv = [ 5,10,15,20 ]
tauv = np.logspace(-2, -4, 3)

for i, tau in enumerate(tauv):
    print(f"Tend/tau {Tend/tau}")

sys.exit()

fig = plt.figure(2)
sv = []
for i, rk in enumerate(rkv):
    erel = []
    for j, tau in enumerate(tauv):
        etime = tm()
        U,S,res = LR_OSI(A, B, X0, Tend, tau, 'rk', rk, verb=0)
        X = U @ S @ U.T
        sv.append(np.diag(S))
        erel.append(rn(X,Xrk))
        print(f'   dt={tau:.0e}:  etime = {tm()-etime:5.2f}   rel error: {rn(X,Xrk):.4e}')
    print('')
    plt.loglog(tauv,erel,label=f'rk={rk:d}',marker='o')
plt.title('Relative error ||X-Xrk||/||Xrk||')
plt.xlabel('dt')
plt.ylabel(f'error vs. RK45({tol:.0e})')
plt.legend()
