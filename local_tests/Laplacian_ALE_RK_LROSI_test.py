#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from time import time as tm
import sys

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from git.solvers.lyapunov import LR_OSI
from git.core.utils import pmat, pvec

def rn(X,Xref):
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

plt.close("all")

eps = 1e-12
n  = 4
I  = np.eye(n)
h  = 1/n**2
At = np.diag(-2/h*np.ones((n,)),0) + \
    np.diag(np.ones((n-1,))/h,1) + \
    np.diag(np.ones((n-1,))/h,-1)
A = np.kron(At,I) + np.kron(I,At)
N = A.shape[0]

# random lhs of rank rk
rk = 5
B = np.zeros((N,rk))
#B = np.random.random_sample((N,rk))
B[:,0] = 1.0

Q = B @ B.T

# random matrix of rank rk = 10
s0    = np.random.random_sample((rk,))
U0, _ = linalg.qr(np.random.random_sample((N, rk)),mode='economic')
S0    = np.diag(s0);

S0 = np.eye(rk)
U0 = np.zeros((N,rk))
for i in range(rk):
    U0[i+1,i] = 1.0

X0    = U0 @ S0 @ U0.T

nQ    = np.linalg.norm(Q)
nA    = np.linalg.norm(A)

# direct Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

# direct solve of A @ X + X @ A.T = -B @ B.T
Xref = linalg.solve_continuous_lyapunov(A, -Q)

pmat(B)
pmat(Q)
pmat(Xref)


sys.exit()
dt = 0.1

U = U0.copy()
S = S0.copy()

pmat(X0, 'X0')

U1 = linalg.expm(dt*A) @ U

UA, R = linalg.qr(U1,mode='economic')
SA    = R @ S @ R.T

X1 = UA @ SA @ UA.T

pmat(X1, 'X1')

# solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
K1 = UA @ SA + dt*(Q @ UA)

# orthonormalise K1
U1, Sh = linalg.qr(K1,mode='economic')

# solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
St = Sh - dt*( U1.T @ Q @ UA )
# solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
#L1  = St @ UA.T + dt*( U1.T @ Q )
L1T  = UA @ St.T + dt*(Q @ U1)
# update S
S1  = L1T.T @ U1

X2 = U1 @ S1 @ U1.T

pmat(X2, 'X2')

Tend = dt
tspan = (0,Tend)
tol = 1e-12
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=tol, rtol=tol)
Xrk = sol.y[:,-1].reshape(A.shape)

pmat(Xrk, 'XRK')



