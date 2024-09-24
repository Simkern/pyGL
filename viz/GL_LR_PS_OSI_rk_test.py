#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, copy
import numpy as np
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_rk_base
from solvers.lyapunov_ProjectorSplittingIntegrator import set_initial_rank, LR_OSI_step, setup_IC, idx_first_below

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

def rn(X,Xref):
    n = Xref.shape[0]
    return np.linalg.norm(X - Xref)/n

plt.close("all")

make_real = True
if_save = True
fldr = 'data_rk'

# Parameters
L0  = 50
x0 = -L0/2                      # beginning of spatial domain
x1 = L0/2                       # end of spatial domain
Nxc = 128
xc = np.linspace(x0, x1, Nxc+2)

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.2
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__   = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xc,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# input and output parameters
rkb = 1
x_b = -11
s_b = 1
rkc = 1
x_c = x12
s_c = 1
rk_X0 = 10

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xc)

# integration weights
dx = np.diff(xc)
wc = np.zeros(Nxc+2)
wc[:Nxc+1] += dx
wc[1:]     += dx

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)
# Input & output
B = np.zeros((Nxc+2, rkb))
B[:,0] = np.exp(-((xc - x_b)/s_b)**2)
C = np.zeros((rkc, Nxc+2))
C[0,:] = np.exp(-((xc - x_c)/s_c)**2)

if make_real:
    # make real
    Lr = np.real(Lc[1:-1,1:-1])
    Li = np.imag(Lc[1:-1,1:-1])
    L  = np.block([[Lr, -Li],[Li, Lr]])
    # Input & output
    Br = np.real(B[1:-1,:])
    Bi = np.imag(B[1:-1,:])
    B = np.block([[Br, -Bi], [Bi, Br]])
    Cr = np.real(C[:,1:-1])
    Ci = np.imag(C[:,:1:-1])
    C = np.block([[Cr, -Ci], [Ci, Cr]])
    # weights and coords
    Nx = 2*Nxc
    w  = np.hstack((wc[1:-1],wc[1:-1]))
    x  = np.hstack((xc[1:-1],xc[1:-1]))
    # plotting prep
    xp = np.hstack((xc[1:-1],xc[1:-1]+L0))
    px,  py  = np.meshgrid(xp,       xp)
    prx, pry = np.meshgrid(xc[1:-1], xc[1:-1])
else:
    L = np.matrix(Lc[1:-1,1:-1])
    w = wc[1:-1]
    x = xc[1:-1]
    # Input & Output
    B = B[1:-1,:]
    C = C[:,1:-1]
    Nx = Nxc
    # plotting prep
    px,  py  = np.meshgrid(x, x)
    prx, pry = np.meshgrid(x, x)

# weight matrix for convenience
W = np.diag(w)

# plotting preparation
box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

# compute controllability gramian

# direct
Qc = B @ B.T @ W
Xref = linalg.solve_continuous_lyapunov(L, -Qc)

Qo = C.T @ C @ W
Yref = linalg.solve_continuous_lyapunov(L.T, -Qo)

nQ    = np.linalg.norm(Qc)
nA    = np.linalg.norm(L)

filename = f'X0_CGL_Nx{Nx:02d}_rk_X0_{rk_X0:02d}.npz'
fname = os.path.join(fldr,filename)
if not os.path.isfile(fname):
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((Nx, rk_X0)),mode='economic')
    S0    = np.diag(sorted(s0)[::-1]);
    X0    = U0 @ S0 @ U0.T
    np.savez(fname, X0=X0, S0=S0, U0=U0)
else:
    data = np.load(fname)
    X0   = data['X0']
    S0   = data['S0']
    U0   = data['U0']
    
U0_svd,S0_svd,V0_svdh = linalg.svd(X0, full_matrices=False)

# compare RK45 to LR_OSI
tol = 1e-12
Trk   = 1
tspan = (0,Trk)
Nrep  = 20
tolv  = np.logspace(-12,-12,1)
Tv    = np.linspace(0,Nrep,Nrep+1)*Trk
filename = f'Xrk_CGL_Nx{Nx:02d}_rk0_{rk_X0:02d}.npz'
fname = os.path.join(fldr,filename)
if not os.path.isfile(fname):
    Xrkv = np.empty((Nx,Nx,len(tolv),Nrep+1))
    erel = np.empty((len(tolv),Nrep+1)) 
    for it, tol in enumerate(tolv):
        X00 = X0
        time = 0.0
        Xrkv[:,:,it,0] = X0
        erel[it,0] = rn(X00,Xref)
        print(f'RK Step {0:2d}, tol={tol:.0e}:  T={time:6.2f}   etime = {0.0:5.2f}   rel error: {rn(X0,Xref):.4e}')
        for i in range(Nrep):
            etime = tm()
            sol = solve_ivp(Xdot,tspan,X00.flatten(),args=(L,Qc), atol=tol, rtol=tol)
            X = sol.y[:,-1].reshape(L.shape)
            Xrkv[:,:,it,i+1] = X
            X00 = X
            erel[it,i+1] = rn(X,Xref)
            time += Trk
            print(f'RK Step {i+1:2d}, tol={tol:.0e}:  T={time:6.2f}   etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.4e}')
        print('')
    np.savez(fname, Xrkv=Xrkv, rel_error_Xref=erel, tolv=tolv, Trk=Trk, Tv=Tv, Nrep=Nrep)
else:
    data = np.load(fname)
    Xrkv = data['Xrkv']
    erel = data['rel_error_Xref']
    Tv   = data['Tv']

nrmX = []
sref = np.zeros((Nrep+1,Nx))
for i in range(Nrep+1):
    nrmX.append(np.linalg.norm(Xrkv[:,:,0,i]))
    _, sref[i,:], _ = np.linalg.svd(Xrkv[:,:,0,i])
        
fig, ax = plt.subplots(1,2)
ax[0].plot(np.linspace(0, Trk*Nrep+1, len(nrmX)), nrmX)
ax[1].semilogy(np.linspace(0, Trk*Nrep+1, len(nrmX)), erel[0,:])

rkv = [ 4, 8, 20, 40, 60]
tauv = np.logspace(0, -3, 7)

fig1, ax1 = plt.subplots(1,2)
p = ax1[0].contourf(px, py, Xref, 100)
fig1.colorbar(p, ax = ax1[0])
ax1[0].set_title('Controllability')

p = ax1[1].contourf(px, py, Yref, 100)
fig1.colorbar(p, ax = ax1[1])
ax1[1].set_title('Observability')

#######################################
#
#  Test initial variation
#
#######################################

torder = 1

for dt in [0.001, 0.01, 0.05, 0.1]:
    exptA = linalg.expm(dt*L)
    for n in range(5,25,5):
        print(f'ninit: {n}, dt = {dt}')
        U, S = set_initial_rank(U0, S0, L, Qc, dt, exptA, torder, verb=0, tol=1e-6, ninit=n)
        
print('\n\n')
Tend = 0.5
for dt in [0.001, 0.01, 0.05, 0.1]:
    exptA = linalg.expm(dt*L)
    n = int(np.ceil(Tend/dt))
    print(f'ninit: {n}, dt = {dt}')
    U, S = set_initial_rank(U0, S0, L, Qc, dt, exptA, torder, verb=0, tol=1e-6, ninit=n)
    
#######################################
#
#  Test dt
#
#######################################

nstep = 100
dtv = [ 0.01, 0.05, 0.1 ]
nt = len(dtv)
tolv = [ 10**(-i) for i in range(6,14,2) ]
ntol = len(tolv)
rkv = np.zeros((nt, nstep, ntol))
    
Tend = 2.0

fig, ax = plt.subplots(1,nt)

for i, dt in enumerate(dtv):
    exptA = linalg.expm(dt*L)
    print(f'dt = {dt}')
    U, S = setup_IC(U0, S0, 60)
    nstep = int(np.ceil(Tend/dt))
    rkv = np.zeros((nstep, ntol))
    for j in range(nstep):
        U, S = LR_OSI_step(U, S, L, Qc, dt, exptA, torder, verb=0)
        _, sv, _ = linalg.svd(S)
        for k, tol in enumerate(tolv):
            rkv[j,k] = idx_first_below(sv, tol)
    for k, tol in enumerate(tolv):
        ax[i].plot(np.linspace(0,Tend,nstep), rkv[:,k], label=f'{tol}')
    ax[i].set_title(f'dt = {dt}')
    ax[i].set_ylim([0,45])
plt.legend()
