#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:28:17 2024

@author: skern
"""

import numpy as np
import time
import sys

from scipy import linalg, sparse
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('./git/core/')

from git.core.utils import pmat, pvec

from git.solvers.lyap_utils import M_ForwardMap, G_ForwardMap

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

plt.close("all")

eps = 1e-12 #3e-14
n  = 4
I  = np.eye(n)
h  = 1/n
h2 = h**2
A  = (np.diag(-2*np.ones((n,)),0) + \
    np.diag(np.ones((n-1,)),1) + \
    np.diag(np.ones((n-1,)),-1))/h2
#A = np.kron(At,I) + np.kron(I,At)
N = A.shape[0]

X0 = np.eye(n) #np.ones_like(A)
S0 = np.eye(n)
B = np.ones((n,1))
Q  = np.ones_like(A)
#Q = np.zeros_like(A)
#Q = B @ B.T
#Q[1,1] = 1

Tend = 0.0011
tspan = (0,Tend)
dt = Tend

etA = linalg.expm(Tend*A)

# RK
etime = time.time()
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=eps, rtol=eps)
Xend = sol.y[:,-1].reshape(A.shape)
print(time.time() - etime)

pmat(X0, 'X0')

exptA = linalg.expm(dt*A)
U1A, S1A = M_ForwardMap(A, X0, X0, dt, exptA)
pmat(U1A @ S1A @ U1A.T, 'X1, Mstep')
U1, S1   = G_ForwardMap(U1A, S1A, Q, dt)
pmat(U1 @ S1 @ U1.T, 'X1, Gstep2')

pmat(Xend, 'Xend RK')


#U0, S0   = G_ForwardMap(U1A, S1A, Q, dt)

sys.exit()

# by hand
dXdt_m = np.zeros_like(X0)
dv  = np.zeros_like(X0)
dvT = np.zeros_like(X0)

X0 = np.random.rand(n,n)

v  = np.copy(X0)
vT = np.copy(X0.T)

for j in range(n):
    #> Left most boundary points 
    #dv[1,i] = (v[-1,i] - 2*v[1,i] + v[1,i]) / h**2 #> Diffusion term (wrap around)
    dv[0,j]  = (- 2*v[0,j] + v[1,j])/h2          #> Diffusion term (zero Dirichlet)
    dvT[j,0] = (- 2*v[j,0] + v[j,1])/h2
    
    #> Interior nodes.
    for i in range(1,n-1):
        dv[ i,j] = (v[i-1,j] - 2*v[i,j] + v[i+1,j])/h2
        dvT[j,i] = (v[j,i-1] - 2*v[j,i] + v[j,i+1])/h2
    
    #> Right most boundary points
    #dv[nx,j] = (v[n-1,j] - 2*v[n,j] + v[1,j]) / h**2 # wrap around
    dv[-1, j] = (v[-2,j] - 2*v[-1,j])/h2
    dvT[j,-1] = (v[j,-2] - 2*v[j,-1])/h2

dXdt_m = (dv + dvT + Q).flatten()

#Xref = linalg.solve_continuous_lyapunov(A, -Q)

E,V = np.linalg.eig(A)

p(np.sort(E)[::-1])


