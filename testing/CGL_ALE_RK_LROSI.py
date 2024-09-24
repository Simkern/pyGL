#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os.path

from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI
from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
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


make_real = True
if_save = True

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
rkX0 = 10

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
    px, py = np.meshgrid(xp, xp)
else:
    L = np.matrix(Lc[1:-1,1:-1])
    w = wc[1:-1]
    x = xc[1:-1]
    # Input & Output
    B = B[1:-1,:]
    C = C[:,1:-1]
    Nx = Nxc
    # plotting prep
    px, py = np.meshgrid(x, x)

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

fig = plt.figure(1)
Uref,Sref,_ = linalg.svd(Xref)
plt.scatter(np.arange(Nx)+1,Sref,s=60,marker='*',label='direct solve')
plt.yscale('log')
plt.xlim(1,90)
plt.ylim(1e-15,100)
plt.yticks([ 10**i for i in range(-14,2,2)])
plt.legend()

fig = plt.figure(1)
Uref,Sref,_ = linalg.svd(Xref)
plt.scatter(np.arange(Nx)+1,Sref,s=60,marker='*',label='direct solve')
plt.yscale('log')
plt.xlim(1,90)
plt.ylim(1e-15,100)
plt.yticks([ 10**i for i in range(-14,2,2)])
plt.legend()

fname = f'../local_tests/CGL_Nx{Nx:02d}_X0.npz'
rk_X0 = 10
if not os.path.isfile(fname):
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((Nx, rk_X0)),mode='economic')
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

Trk   = 0.1
tspan = (0,Trk)
Nrep  = 800
tolv  = np.logspace(-12,-12,1)
Tv    = np.linspace(0,Nrep,Nrep+1)*Trk
fname = f'../local_tests/CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_Xrk.npz'
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
    plt.scatter(np.arange(Nx)+1,Srk_svd,
                   s=20,
                   label=f'S_rk, T={t:.2e}')
    
plt.scatter(np.arange(Nx)+1,Sref,s=40,
            marker='*',
            color='k',
            label='direct solve')
plt.xlim(0,20)
plt.ylim(1e-7,1000)
plt.yticks([ 10**i for i in range(-4,2,2)])
plt.yscale('log')
plt.legend()

# compare RK45 to LR_OSI
Tend = 60
tspan = (0,Tend)
tol = 1e-12
Xrk = np.squeeze(Xrkv[:,:,tolv==tol,Tv==Tend])
Urk_svd,Srk_svd,Vrk_svdh = linalg.svd(Xrk, full_matrices=False)
print(f'Tend = {Tend:6.2f}, RK error = {rn(Xrk, Xref):4e}')

#rkv = [2, 8, 14, 20]
rkv = [8, 20, 40]
#rkv = np.arange(2,18+1,4)
#tauv = np.logspace(-1, -5, 5)
tauv = [ 0.1, 0.01, 0.001 ]
tord = [ 1, 2 ]

#fig, axs = plt.subplots(1,len(tauv)+1)
#plotX = axs[0].contourf(px, py, Xrk, 100)
##fig.colorbar(plotX, ax = axs[0])
#axs[0].set_title('RK')

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
            U, S = LR_OSI(U0_rk, S0_rk, L, Qc, Tend, dt, torder=torder, verb=0)
            X = U @ S @ U.T
            u,s,_ = linalg.svd(X)
            Sdlra[:rk,i,j,it]   = s[:rk]
            edlra[    i,j,it] = rn(X,Xrk)
            print(f'\tdt={dt:.0e}:  etime = {tm()-etime:5.2f}   err RK: {rn(X,Xrk):.4e} err ref: {rn(X,Xref):.4e}')
            #plotX = axs[j+1].contourf(px, py, X-Xrk, 100)
            #fig.colorbar(plotX, ax = axs[j+1])
            #axs[j+1].set_title(f'dt = {dt:.5f}')

fig, axs = plt.subplots(1,3)
for i, rk in enumerate(rkv):
    axs[0].loglog(tauv,edlra[i,:,0],label=f'rk={rk:d}, torder=1',marker='o')

if len(tord) > 1:
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
axs[1].scatter(np.arange(Nx)+1,Srk_svd,
               s=20,
               color='r',
               label='S_rk')

for i, rk in enumerate(rkv):
    it = 1
    tau = tauv[it]
    axs[2].scatter(np.arange(rk)+1,Sdlra[:rk,i,it,0],
                   s=40,
                   marker='o',
                   #facecolors='none',
                   label=f'T = {Tend:4.1f}, rk = {rk:2d}, tau={tau:.1e}')

plt.scatter(np.arange(Nx)+1,Sref,s=40,
            marker='*',
            color='k',
            label='direct solve')
axs[2].set_xlim(0,20)
axs[2].set_ylim(1e-7,1000)
axs[2].set_yticks([ 10**i for i in range(-4,2,2)])
axs[2].set_yscale('log')
axs[2].legend()

