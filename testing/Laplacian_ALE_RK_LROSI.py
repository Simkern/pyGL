#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from git.solvers.lyapunov import LR_OSI

def rn(X,Xref):
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

plt.close("all")

eps = 1e-12
n  = 20
I  = np.eye(n)
h  = 1/(n+1)
At = np.diag(-2/h*np.ones((n,)),0) + \
    np.diag(np.ones((n-1,))/h,1) + \
    np.diag(np.ones((n-1,))/h,-1)
A = np.kron(At,I) + np.kron(I,At)
N = A.shape[0]

# random lhs of rank rk
rk = 5
B = np.random.random_sample((N,rk))

Q = B @ B.T

# random matrix of rank rk = 10
s0    = np.random.random_sample((rk,))
U0, _ = linalg.qr(np.random.random_sample((N, rk)),mode='economic')
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

etime = tm()
# direct solve of A @ X + X @ A.T = -B @ B.T
Xref = linalg.solve_continuous_lyapunov(A, -Q)
print(f'Direct solve:    etime = {tm()-etime:5.2f}')

Tend = 4 #5  #4
tspan = (0,Tend)
Nrep = 5 #10 #5
tolv = np.logspace(-6,-12,4)

Xa = np.empty((N,N,len(tolv),Nrep))

fig = plt.figure(1)
for it, tol in enumerate(tolv):
    X00 = X0
    erel = [rn(X00,Xref)]
    for i in range(Nrep):
        etime = tm()
        sol = solve_ivp(Xdot,tspan,X00.flatten(),args=(A,Q), atol=tol, rtol=tol)
        X = sol.y[:,-1].reshape(A.shape)
        Xa[:,:,it,i] = X
        X00 = X
        erel.append(rn(X,Xref))
        print(f'RK Step {i+1:2d}, tol={tol:.0e}:   etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.4e}')
    print('')
    plt.semilogy(np.arange(Nrep+1)*Tend,erel,label=f'tol={tol:.0e}',marker='o')
plt.title('Relative error ||X-Xref||/||Xref||')
plt.xlabel('Integration time [s]')
plt.ylabel('error vs. direct solve')
plt.legend()

# compare RK45 to LR_OSI
Tend = 0.1
tspan = (0,Tend)
tol = 1e-12
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=tol, rtol=tol)
Xrk = sol.y[:,-1].reshape(A.shape)
Urk,Srk,Vrkh = linalg.svd(Xrk)

rkv = [ 5,10,15,20 ]
tauv = np.logspace(-2, -5, 10)

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
