#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:28:17 2024

@author: skern
"""

import numpy as np
from time import time as tm
import sys

from scipy import linalg, sparse
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('./git/core/')

from git.core.utils import pmat, pvec
from git.solvers.arnoldi import *

from git.solvers.lyap_utils import M_ForwardMap, G_ForwardMap, kryl_expm
from git.solvers.lyapunov import LR_OSI

def rn(X,Xref):
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

plt.close("all")

eps = 1e-12 #3e-14
n  = 4
rk = 5

I  = np.eye(n)
h  = 1/n
h2 = h**2
At  = (np.diag(-2*np.ones((n,)),0) + \
    np.diag(np.ones((n-1,)),1) + \
    np.diag(np.ones((n-1,)),-1))/h2
A = np.kron(At,I) + np.kron(I,At)
N = A.shape[0]


#B = np.zeros((N,1))
#B[0] = 1
B = np.ones((N,1))
#Q  = np.ones_like(A)
#B = np.random.rand(N,rk)
Q = B @ B.T

Tend = 10.0
tstep = 1.0
tspan = (0,Tend)

'''
# RK
#time = time.time()
print(f'Reference: Tend = {Tend:.2f}')
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=eps, rtol=eps)
yref = sol.y[:,-1]

for it, te in enumerate(np.arange(tstep,2.0,tstep)):
    tspan = (0,te)
    sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=eps, rtol=eps)
    y = sol.y[:,-1]
    print(f'Step {it+1:d}: Tend = {te:.2f}, ||err||_2 = {linalg.norm(y - yref):16.7e}')
'''
etime = tm()
# direct solve of A @ X + X @ A.T = -B @ B.T
Xref = linalg.solve_continuous_lyapunov(A, -Q)
print(f'Direct solve:    etime = {tm()-etime:5.2f}')

#pmat(Xref, 'Xref')

rk0 = 5

X0 = np.random.rand(N,rk0)
#X0 = np.zeros((N,rk0))
X0[:rk0,:rk0] = np.eye(rk0)

U00,S00,_ = linalg.svd(X0, full_matrices=False)


rkv = [ 2, 8, 14 ]
#rkv = [ 2, 6, 10, 14 ]
#rkv = [ 2, 8, 14 ]
tauv = [ 1e-1, 1e-3, 1e-4, 5e-5 ]
#tauv = np.logspace(-2, -5, 3)

#B = np.zeros((N,1))
#B = np.ones((N,1))

pmat(U00, 'U0')

Tend = 1

sv = []
for i, rk in enumerate(rkv):
    U = np.zeros((N,rk))
    U[:,:min(rk,rk0)] = U00[:,:min(rk,rk0)]
    Ss = np.zeros((rk,))
    Ss[:min(rk,rk0)] = S00[:min(rk,rk0)]
    S = np.diag(Ss)
    X00 = U @ S @ U.T
    '''
    pmat(U, 'U0')
    pmat(S, 'S0')
    pmat(B, 'B')
    '''
   
    for j, tau in enumerate(tauv):
        '''
        nt    = int(np.ceil(Tend/tau))
        dt    = Tend/nt
        tspan = np.linspace(0, Tend, num=nt, endpoint=True)
        X0 = X00
        for k in range(nt):
            etime = tm()
            U,S,res = LR_OSI(A, B, X0, dt, dt, 'rk', rk, verb=0)
            X = U @ S @ U.T
            X0 = X.copy()
            #pmat(U,'U')
            #pmat(S,'S')
            sv.append(np.diag(S))
            #pmat(U@S@U.T,'USUT')
            print(f'   dt={tau:.0e}:  Tend = {Tend:.0e}, etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.8e}')
        '''
        etime = tm()
        #print(f'Tend = {Tend}, tau = {tau}, n,N = {n},{N}')
        '''#
        U1 = kryl_expm(A,U,15,tau)
        #pmat(U1, 'M U1')
        UA, R, P = linalg.qr(U1,mode='economic', pivoting=True)
        #print(R==0)
        SA    = R @ S @ R.T

        #print(SA==0)
        #pmat(UA, 'M UA')
        #pmat(SA, 'M SA')
        
        # solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
        K1 = UA @ SA + tau*(Q @ UA)
            
        #pmat((Q @ UA), 'G Q@UA')
        #pmat(K1, 'G K1')
        
        # orthonormalise K1
        U1, Sh, P = linalg.qr(K1,mode='economic', pivoting=True)
            
        #pmat(U1, 'G U1')
        #pmat(Sh, 'G Sh')
        #for i in range(rk):
        #    print(f' {Sh[i,i]:.2e}', end = '')
        #print(Sh==0)
        #print('Sh')
        #sys.exit()
        # solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
        St = Sh - tau*( U1.T @ Q @ UA )
        pmat(St, 'G St')
        # solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
        L1  = St @ UA.T + tau*( U1.T @ Q )
        # update S
        S1  = L1 @ U1
        
        pmat(S1,'S1')
        '''
        U,S,res = LR_OSI(A, B, X00, Tend, tau, 'rk', rk, verb=0)
        X = U @ S @ U.T
        print(f'rk = {rk:3d}   dt={tau:.0e}:  Tend = {Tend:.0e}, etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.8e}')
    print('')