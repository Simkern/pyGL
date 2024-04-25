#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from time import time as tm
import sys

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from git.core.CGL_parameters import CGL, CGL2
from git.core.diff_mat import FDmat
from git.core.utils import p as prnt
from git.solvers.lyapunov import LR_OSI
from git.solvers.lyap_utils import kryl_expm
from git.solvers.arnoldi import arn

def rn(X,Xref):
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

plt.close("all")

Nx = 20
A = np.random.rand(Nx,Nx)

# random lhs of rank rk
rk = 2 # 10
B = np.random.random_sample((Nx,rk))
Qb,Rb = linalg.qr(B,mode='economic')

dt = 1

'''
nkryl = 1 #20

Q_, H_ = arn(A,B,nkryl,0)

Q2, H2 = arn(A,Qb,nkryl,0)

p = rk*nkryl

Q = Q2[:,:p]
H = H2[:p,:p]

exptH = linalg.expm(dt*H)

X = (Q @ exptH[:,:rk]) @ Rb
'''
'''
ai = np.empty([])
for i in range(int(Nx/2)):
    ai = np.append(ai, [(2*(i+1) - 1)/(Nx+1), (2*(i+1) - 1)/(Nx+1)])
c = np.empty([])
for i in range(int(Nx/2)):
    c = np.append(c, [0.5, 0.0])
A = np.diag(ai[1:]) + np.diag(-c[1:-1],k=-1) + np.diag(c[1:-1],k=1)

exptA = linalg.expm(A)
Xref = np.ones([Nx,1])
B = linalg.solve(exptA,Xref)
'''

Qb,Rb = linalg.qr(B,mode='economic')

Xref = linalg.expm(A) @ B

Anorm = np.linalg.norm(dt*A)

for nkryl in range(1,6): #range(1,20):
    X, Y, X2, e = kryl_expm(A,B,nkryl,dt,Anorm)
    
    X_ = X + Y

    e1 = np.linalg.norm(X  - Xref)
    e2 = np.linalg.norm(X_ - Xref)
    e3 = np.linalg.norm(X2 - Xref)
    print(' m = {n:2d}, e1 = {e1:e}, e2 = {e2:e}, e3 = {e3:e}, estimate = {e:e}'.format(n=nkryl,e1=e1, e2=e2, e3=e3, e=e))
    #print(' m = {n:2d}, e1 = {e1:e}, e2 = {e2:e}'.format(n=nkryl,e1=e1, e2=e2))


'''
p = B.shape[1]
H = np.zeros((p*(n+1), p*n), dtype=A.dtype)
Q = np.zeros((A.shape[0], p*(n+1)), dtype=A.dtype)
test = np.arange(48).reshape(H.shape)
# Ortho-normalize the input matrix
Q[:, :p], _ = linalg.qr(B, mode='economic')  # Use it as the first Krylov basis
for k in range(n):
    print(f'  {k:3d}')
    s   = slice(p*k,p*(k+1))
    sp1 = slice(p*(k+1),p*(k+2))
    V = A @ Q[:,s]  # Generate a new candidate matrix
    # for each colum of Q that we have already constructed
    for l in range(2):      # MGS
        k_min = max(0, k - n - 1)
        for ll in range(k_min, k + 1):
            sl       = slice(p*ll,p*(ll+1))
            proj     = Q[:,sl].conj().T @ V
            H[sl,s] += proj
            V       -= Q[:,sl] @ proj
    # Ortho-normalize result
    Q[:,sp1], H[sp1,s] = linalg.qr(V, mode='economic')
'''