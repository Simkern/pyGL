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

sys.path.append('..')

from git.core.CGL_parameters import CGL, CGL2
from git.core.diff_mat import FDmat

from git.solvers.arnoldi import arn, arn_inv
from git.solvers.lyapunov import lrcfadic_r_gmres

from CGLmat import getCGLmatrices
from wrappers import gmres_wrapper, pgmres_lu, cg_wrapper, pgmres_cg

plt.close("all")

A,L,Lc,Nx,Nxc = getCGLmatrices()

def ydotCGL(t, y):
    global A
    return A @ y

def forwardMap(v, Tend=1):  
    sol = solve_ivp(ydotCGL, (0,Tend), v)
    return sol.y[:,-1]

E,V = np.linalg.eig(A)

b = np.zeros((Nx,))
b[1] = 1

tspace = np.logspace(-1,1,5)
nvec = [ 10, 30, 60 ]

for i, n in enumerate(nvec):
    fig = plt.figure(2*i+1)
    ax1 = fig.add_subplot(1, 1, 1) 
    fig = plt.figure(2*i+2)
    ax2 = fig.add_subplot(1, 1, 1) 
    ax2.scatter(np.real(E),np.imag(E),marker='x',color='k')

    for Tend in tspace:
        exponentialPropagator = LinearOperator(matvec = lambda v: forwardMap(v, Tend),
                                               shape=(Nx,Nx))
        Q, H = arn(exponentialPropagator, b, n)
        Eh = np.linalg.eig(H[:n,:n])[0]
        Ea = np.log(Eh)/Tend
        ax1.scatter(np.real(Eh),np.imag(Eh),label=f'T = {Tend:3.2f"}')
        ax2.scatter(np.real(Ea),np.imag(Ea),label=f'T = {Tend:3.2f}')

    ax1.legend()
    ax2.legend()    
    ax1.scatter(1,0,marker='x',color='k')
    ax2.set_xlim(-350, 20)
    ax2.set_ylim(-420,420)
    ax1.set_title(f'eig(exp(T A)), n = {n}')
    ax2.set_title(f'eig(A), n = {n}')