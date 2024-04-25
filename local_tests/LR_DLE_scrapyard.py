#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:24:25 2024

@author: skern
"""

import numpy as np
import time
import sys

from scipy import linalg, sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, gmres, cg
from matplotlib import pyplot as plt

sys.path.append('./git/core/')

from git.core.CGL_parameters import CGL, CGL2
from git.core.diff_mat import FDmat
from git.core.utils import pmat, pvec, p

from git.solvers.arnoldi import arn, arn_inv
from git.solvers.lyapunov import kpik

from CGLmat import getCGLmatrices
from wrappers import gmres_wrapper, pgmres_lu, cg_wrapper, pgmres_cg

def Mdot(t, m, A):
    """
    dM/dt = A @ M + M @ A^T
    """
    
    n    = A.shape[0]
    M    = m.reshape((n,-1))
    dMdt = A @ M + M @ A.T
    
    return dMdt.flatten()

def Kdot(t, k, Q, U):

    dKdt = Q @ U
    
    return dKdt.flatten()

def Sdot(t, s, Q, U, UA):
    
    dSdt = - U.T @ Q @ UA
    
    return dSdt.flatten()

def Ldot(t, l, Q, U):

    dLdt = U.T @ Q
    
    return dLdt.flatten()

def M_mexpSymForwardMap(A,U,S,tau,exptA=None):
    
    if exptA is None:
        U1    = linalg.expm(tau*A) @ U
    else:
        U1    = exptA @ U
    UA, R = linalg.qr(U1,mode='economic')

    SA    = R @ S @ R.T
    
    return UA, SA

def G_ForwardMap(UA, SA, Q, tau):
    
    n,r = UA.shape
    
    # solve Kdot = Q @ U1A with K0 = UA @ SA
    K0 = UA @ SA
    sol = solve_ivp(Kdot, (0,tau), K0.flatten(), args=(Q, UA))
    
    K1 = sol.y[:,-1].reshape((n,r))
    # orthonormalise K
    U1, Sh = linalg.qr(K1,mode='economic')
    
    # solve Sdot = - U1.T @ Q @ UA
    sol = solve_ivp(Sdot, (0,tau), Sh.flatten(), args=(Q, U1, UA))
    St = sol.y[:,-1].reshape((r,r))
    
    # solve Ldot = U1.T @ Q
    L0 = St @ UA.T
    sol = solve_ivp(Ldot, (0,tau), L0.flatten(), args=(Q, U1))   
    L1  = sol.y[:,-1].reshape((r,n))
    
    S1  = L1 @ U1
   
    return U1, S1

def ALE_integrate(A,B,dt,n,U0,S0):
    # time-integration of dAdt = A @ X + X @ A.T + Q
    
    Q = B @ B.T
    
    res = []
    
    for i in range(n):
        
        print(f'Step {i}; t = {(i+1)*dt:3.2f}')
        
        U1A, S1A = M_mexpSymForwardMap(A, U0, S0, dt)
        
        U0, S0   = G_ForwardMap(U1A, S1A, Q, dt)
        
        X = U0 @ S0 @ U0.T
        
        res.append(np.linalg.norm(A @ X + X @ A.T + Q))
        
    return X, res

def Xdot(t,Xv,A,Q):
    
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    
    return dXdt.flatten()
        
plt.close("all")

A,L,Lc,Nx,Nxc = getCGLmatrices()
B = np.random.rand(Nx,1)

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

nQ    = np.linalg.norm(Q)
nA    = np.linalg.norm(A)

# random matrix of rank rk
rk = 10
# choose eigenvalues
s0 = np.random.random_sample((rk,))
# full up with 0+ (for SPD)
#z0 = eps*np.ones(N-rk,)
#e0 = np.concatenate((s0,z0))
# generate eigenvectors
U0, _ = linalg.qr(np.random.random_sample((N, rk)),mode='economic')
S0 = np.diag(s0);
#U     = np.random.rand(N,rk)
#X     = U @ U.T
#S0, U0 = np.linalg.eig(X)
#U0, R = linalg.qr(U,mode='economic')
#s0    = [ abs(s) for s in np.diag(R) ]
#S0    = np.diag(s0)
# for reference 
X0    = U0 @ S0 @ U0.T

# direct solve of A @ X + X @ A.T = -B @ B.T
Xref = linalg.solve_continuous_lyapunov(A, -Q)

Tend = 0.1
nt = 5
t_eval = np.linspace(0,Tend,nt)
sol = solve_ivp(Xdot, (0,Tend), X0.flatten(), args=(A,Q), t_eval = t_eval)

res = []
Xt  = np.empty((N,N,nt)) 
for i in range(nt):
    Xt[:,:,i] = sol.y[:,i].reshape(A.shape)
    nX = np.linalg.norm(Xt[:,:,i])
    res.append(np.linalg.norm(Xref - Xt[:,:,i])/(nQ + nA*nX))

fig = plt.figure(1)
ax = fig.add_subplot(121)

Uref,Sref,Vref = np.linalg.svd(Xref)
for i in range(nt):
    U1,S1,V1 = np.linalg.svd(Xt[:,:,i])
    if i < nt-1:
        ax.semilogy(S1,'o', label=f'T = {t_eval[i]:4.3f}')
plt.semilogy(S1,'kx', label=f'T = {t_eval[-1]}')
plt.semilogy(Sref,'k*', label='Steady state Xref')
plt.xlim(0,90)
plt.legend()
plt.title('Singular values of X (RK5 vs direct)')

ax = fig.add_subplot(122)
ax.semilogy(sol.t,res)
plt.title('|| X - Xref ||/(||Q|| + ||A||*||X||)')

'''
tolY = 1e-12
tol  = 1e-10
k_max = 100
Z, err2, k_eff, etime = kpik(A,B,k_max,tol,tolY)
nstep = err2.shape[0]
#print(' its |        comp.res. | space dim. |       CPU Time')
#print(f'{nstep:4d} |   {err2[-1]:14.12f} |       {k_eff:4d} | {etime:14.12f}')

X = Z @ Z.T
nX = np.linalg.norm(X)
print(np.linalg.norm(X - Xref)/(nQ + nX*nA))
'''

dtaim = 0.01
n     = int(np.ceil(Tend/dtaim))
dt    = Tend/n

exptA = linalg.expm(dt*A)

fig = plt.figure(3)

for rk in [5, 10, 20, N]:
    # random matrix of rank rk
    #rk = 10
    s0 = np.random.random_sample((rk,))
    U0, _ = linalg.qr(np.random.random_sample((N, rk)),mode='economic')
    S0 = np.diag(s0);
    X0    = U0 @ S0 @ U0.T
    
    res   = []
    dif   = []
    dXdt  = []
    
    for i in range(n):
        
        print(f'Step {i}; t = {(i+1)*dt:04.2f},  rk = {rk:2d}')
        Xold     = U0 @ S0 @ U0.T
        
        U1A, S1A = M_mexpSymForwardMap(A, U0, S0, dt, exptA)
        
        U0, S0   = G_ForwardMap(U1A, S1A, Q, dt)
        
        X = U0 @ S0 @ U0.T
        
        res.append(np.linalg.norm(A @ X + X @ A.T + Q)/nQ)
        dif.append(np.linalg.norm(X - Xref)/nQ)
        dXdt.append(np.linalg.norm(X - Xold)/dt)

    U,S,V = np.linalg.svd(X)
    
    plt.semilogy(S,'x',label=f'rk = {rk:d}')
    
plt.semilogy(Sref,'k*', label='Steady state Xref (direct BS solver)')
plt.xlim(0,90)
plt.title('Singular values of X (split-step integrator)')
plt.legend()

# fig = plt.figure(2)
# ax = fig.add_subplot(131)
# plt.plot(res)
# plt.title('|| AX + XAT + Q||/||Q||')
# fig.add_subplot(132)
# plt.plot(dif)
# plt.title('|| X - Xref ||/||Q||')
# fig.add_subplot(133)
# plt.semilogy(dXdt)
# plt.title('|| Xdot ||')

sys.exit()

Q = B @ B.T   

dtaim = 0.001
Tend  = 1
n     = int(np.ceil(Tend/dtaim))
dt    = Tend/n

U0    = np.zeros((Nx,10))
S0    = np.eye(10)

U1    = linalg.expm(dt*A) @ U0
UA, R = linalg.qr(U1,mode='economic')

SA    = R @ S0 @ R.T
'''
X, res = ALE_integrate(A,B,dt,n,U0,S0)

fig = plt.figure(1)
plt.plot(res)


def Ndot(t, N, A):
    
    dNdt = A.flatten()
    
    return dNdt

N0 = np.eye(4)

A = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])

tau = 2
args = (A,)

sol = solve_ivp(Ndot, (0,tau), N0.flatten(), args=args)
'''