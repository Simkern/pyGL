#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os.path

from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_base as LR_OSI
from core.utils import p

def rn(X,Xref):
    n = Xref.shape[0]
    return np.linalg.norm(X - Xref)/n

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

plt.close("all")

eps = 1e-12
n  = 10
I  = np.eye(n)
h  = 1/n**2
At = np.diag(-2/h*np.ones((n,)),0) + \
    np.diag(np.ones((n-1,))/h,1) + \
    np.diag(np.ones((n-1,))/h,-1)
A = np.kron(At,I) + np.kron(I,At)
N = A.shape[0]

# random lhs of rank rk
rk_b = 5

# random matrix of rank rk = 10
rk_X0 = 10

fname = f'../local_tests/Laplacian_n{n:02d}_Xref.npz'
if not os.path.isfile(fname):
    B  = np.random.random_sample((N,rk_b))
    Q  = B @ B.T
    etime = tm()
    # direct solve of A @ X + X @ A.T = -B @ B.T
    Xref = linalg.solve_continuous_lyapunov(A, -Q)
    print(f'Direct solve:    etime = {tm()-etime:5.2f}')
    np.savez(fname, Xref=Xref, A=A, Q=Q)
else:
    data = np.load(fname)
    Xref = data['Xref']
    A = data['A']
    Q = data['Q']

fig = plt.figure(1)
Uref,Sref,_ = linalg.svd(Xref)
plt.scatter(np.arange(N)+1,Sref,s=60,marker='*',label='direct solve')
plt.yscale('log')
plt.xlim(1,90)
plt.ylim(1e-15,100)
plt.yticks([ 10**i for i in range(-14,2,2)])
plt.legend()

fname = f'../local_tests/Laplacian_n{n:02d}_X0.npz'
if not os.path.isfile(fname):
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((N, rk_X0)),mode='economic')
    S0    = np.diag(sorted(s0)[::-1]);
    X0    = U0 @ S0 @ U0.T
    np.savez(fname, X0=X0, S0=S0, U0=U0, rk_X0=rk_X0)
else:
    data = np.load(fname)
    X0   = data['X0']
    S0   = data['S0']
    U0   = data['U0']
    rk_X0= data['rk_X0']
U0_svd,S0_svd,V0_svdh = linalg.svd(X0, full_matrices=False)

Trk   = 0.05
tspan = (0,Trk)
Nrep  = 20
tolv  = np.logspace(-6,-12,4)
Tv    = np.linspace(0,Nrep,Nrep+1)*Trk
fname = f'../local_tests/Laplacian_n{n:02d}_Xrk.npz'
if not os.path.isfile(fname):
    Xrkv = np.empty((N,N,len(tolv),Nrep+1))
    erel = np.empty((len(tolv),Nrep+1)) 
    for it, tol in enumerate(tolv):
        X00 = X0
        Xrkv[:,:,it,0] = X0
        erel[it,0] = rn(X00,Xref)
        print(f'RK Step {0:2d}, tol={tol:.0e}:   etime = {0.0:5.2f}   rel error: {rn(X0,Xref):.4e}')
        for i in range(Nrep):
            etime = tm()
            sol = solve_ivp(Xdot,tspan,X00.flatten(),args=(A,Q), atol=tol, rtol=tol)
            X = sol.y[:,-1].reshape(A.shape)
            Xrkv[:,:,it,i+1] = X
            X00 = X
            erel[it,i+1] = rn(X,Xref)
            print(f'RK Step {i+1:2d}, tol={tol:.0e}:   etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.4e}')
        print('')
    np.savez(fname, Xrkv=Xrkv, rel_error_Xref=erel, tolv=tolv, Trk=Trk, Tv=Tv, Nrep=Nrep)
else:
    data = np.load(fname)
    Xrkv = data['Xrkv']
    erel = data['rel_error_Xref']
    Tv   = data['Tv']
    
fig = plt.figure(2)
for it, tol in enumerate(tolv):
    plt.semilogy(np.arange(Nrep+1)*Trk,erel[it,:],label=f'tol={tol:.0e}',marker='o')
plt.title('Relative error ||X-Xref||/N')
plt.xlabel('Integration time [s]')
plt.ylabel('error vs. direct solve')
plt.legend()

fig = plt.figure(3)
plt.scatter(np.arange(rk_X0)+1,S0_svd[:rk_X0],
               s=10,
               color='k',
               label='S0')
tol = 1e-12
for it, t in enumerate(Tv[0:-1:4]):
    Xrk = np.squeeze(Xrkv[:,:,tolv==tol,it])
    Urk_svd,Srk_svd,Vrk_svdh = linalg.svd(Xrk, full_matrices=False)
    plt.scatter(np.arange(N)+1,Srk_svd,
                   s=20,
                   label=f'S_rk, T={t:.2e}')
    
plt.scatter(np.arange(N)+1,Sref,s=40,
            marker='*',
            color='k',
            label='direct solve')
plt.xlim(0,20)
plt.ylim(1e-7,100)
plt.yticks([ 10**i for i in range(-4,2,2)])
plt.yscale('log')
plt.legend()

# compare RK45 to LR_OSI
Tend = 0.1
tspan = (0,Tend)
tol = 1e-12
Xrk = np.squeeze(Xrkv[:,:,tolv==tol,Tv==Tend])
Urk_svd,Srk_svd,Vrk_svdh = linalg.svd(Xrk, full_matrices=False)

rkv = [2, 8, 14, 20, 30]
#rkv = np.arange(2,18+1,4)
tauv = np.logspace(-1, -5, 5)
tord = [ 1, 2 ]

rkmax = np.max(rkv)
Sdlra = np.empty((rkmax, len(rkv), len(tauv), len(tord)))
Sdlra.fill(np.NaN)
edlra = np.empty((       len(rkv), len(tauv), len(tord)))
for it, torder in enumerate(tord):
    for i, rk in enumerate(rkv):
        U0_rk = U0_svd[:,:rk]
        S0_rk = np.diag(S0_svd[:rk])
        print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
        for j, dt in enumerate(tauv):
            etime = tm()
            U, S = LR_OSI(U0_rk, S0_rk, A, Q, Tend, dt, torder=torder, verb=0)
            X = U @ S @ U.T
            u,s,_ = linalg.svd(X)
            Sdlra[:rk,i,j,it]   = s[:rk]
            edlra[    i,j,it] = rn(X,Xrk)
            print(f'\tdt={dt:.0e}:  etime = {tm()-etime:5.2f}   rel error: {rn(X,Xrk):.4e}')

fig, axs = plt.subplots(1,3)
for i, rk in enumerate(rkv):
    axs[0].loglog(tauv,edlra[i,:,0],label=f'rk={rk:d}, torder=1',marker='o')

for i, rk in enumerate(rkv):
    axs[0].loglog(tauv,edlra[i,:,1],label=f'rk={rk:d}, torder=2',marker='o',linestyle='--')
axs[0].set_title('||X-Xrk||/N (order 2)')
axs[0].set_xlabel('dt')
axs[0].set_ylabel(f'error vs. RK45({tol:.0e})')
axs[0].set_ylim(1e-8,1)
axs[0].legend()

axs[1].scatter(np.arange(rk_X0)+1,S0_svd[:rk_X0],
               s=10,
               color='k',
               label='S0')
axs[1].scatter(np.arange(N)+1,Srk_svd,
               s=20,
               color='r',
               label='S_rk')
axs[1].set_xlim(0,20)
axs[1].set_ylim(1e-7,100)
axs[1].set_yticks([ 10**i for i in range(-4,2,2)])
axs[1].set_yscale('log')
axs[1].legend()

for i, rk in enumerate(rkv):
    it = 3
    tau = tauv[it]
    axs[2].scatter(np.arange(rk)+1,Sdlra[:rk,i,it,0],
                   s=40,
                   marker='o',
                   #facecolors='none',
                   label=f'T = {Tend:4.1f}, rk = {rk:2d}, tau={tau:.1e}')

plt.scatter(np.arange(N)+1,Sref,s=40,
            marker='*',
            color='k',
            label='direct solve')
axs[2].set_xlim(0,20)
axs[2].set_ylim(1e-7,100)
axs[2].set_yticks([ 10**i for i in range(-4,2,2)])
axs[2].set_yscale('log')
axs[2].legend()

