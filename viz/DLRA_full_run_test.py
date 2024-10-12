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

from solvers.lyap_utils import CALE
from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_test

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
fldr = 'data'

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
wc = np.ones(Nxc+2)*dx[0]

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)
# Input & output
B = np.zeros((Nxc+2, rkb)); B[:,0] = np.exp(-((xc - x_b)/s_b)**2)
C = np.zeros((rkc, Nxc+2)); C[0,:] = np.exp(-((xc - x_c)/s_c)**2)

if make_real:
    # make real
    Lr = np.real(Lc[1:-1,1:-1]); Li = np.imag(Lc[1:-1,1:-1])
    L  = np.block([[Lr, -Li],[Li, Lr]])
    # Input & output
    Br = np.real(B[1:-1,:]); Bi = np.imag(B[1:-1,:])
    B = np.block([[Br, -Bi], [Bi, Br]])
    Cr = np.real(C[:,1:-1]); Ci = np.imag(C[:,:1:-1])
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

filenameU = f'CGL_Nx{Nx:02d}_U0_rk_X0_{rk_X0:02d}.npy'
filenameS = f'CGL_Nx{Nx:02d}_S0_rk_X0_{rk_X0:02d}.npy'
fnameU = os.path.join(fldr,filenameU)
fnameS = os.path.join(fldr,filenameS)
if not os.path.isfile(fnameU):
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((Nx, rk_X0)),mode='economic')
    S0    = np.diag(sorted(s0)[::-1]);
    np.save(fnameU, U0)
    np.save(fnameS, S0)
else:
    U0 = np.load(fnameU)
    S0 = np.load(fnameS)
    
X0    = U0 @ S0 @ U0.T
U0_svd,S0_svd,V0_svdh = linalg.svd(X0, full_matrices=False)

# compare RK45 to LR_OSI
tol = 1e-12
Trk   = 1
tspan = (0,Trk)
Nrep  = 100
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
  
rkv = [ 4, 20 ] #[ 4, 8, 20, 40, 60]
tauv = np.logspace(0, -3, 4)

# compare RK45 to LR_OSI
ifexpm = True
TOv = [1, 2]
Tend = 1
tspan = (0,Tend)
tol = 1e-12
Xrk = np.squeeze(Xrkv[:,:,tolv==tol,Tv==Tend])
Urk_svd,Srk_svd,Vrk_svdh = linalg.svd(Xrk, full_matrices=False)
errRK = rn(Xrk, Xref)
print(f'T0   = {0.0:6.2f}, RK error = {rn(X0, Xref):4e}, res_0  = {CALE(X0,L,Qc):4e}')
print(f'Tend = {Tend:6.2f}, RK error = {errRK:4e}, res_RK = {CALE(Xrk,L,Qc):4e}')
print(f'|X_RK|/N = {np.linalg.norm(Xrk)/Nx:4e}')

print('\nInitial condition: X0')
print(f'    |res| = {CALE(X0,L,Qc):4e}')
print('    SVD = ',' '.join(f'{x:2e}' for x in linalg.svdvals(X0)[:10]))

fig, axs = plt.subplots(len(rkv),2)
erk = np.zeros((len(TOv), len(rkv), len(tauv)))
erf = np.zeros((len(TOv), len(rkv), len(tauv)))
nrm = np.zeros((len(TOv), len(rkv), len(tauv)))
for i, rk in enumerate(rkv):
    for it, torder in enumerate(TOv):
        print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
        for j, tau in enumerate(tauv):
            #print(f' dt={tau:.0e}:')
            filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_rk_{rk:02d}_dt{tau:.2e}_TO{torder:1d}_T_{Tend:.2e}.npz'
            fname = os.path.join(fldr,filename)
            if not os.path.isfile(fname):
                etime = tm()
                nt = int(np.floor(Tend/tau))
                X00 = copy.deepcopy(X0)
                U, S, svals, res_rk, res_rf = LR_OSI_test(L, Qc, X0, Xrk, Xref, Tend, tau, rk, torder=torder, verb=0, ifexpm=ifexpm)
                #print(f'size: {U.shape} {S.shape}')
                X = U @ S @ U.T
                #np.savez(fname, X=X, U=U, S=S, X0=X0, Qc=Qc, Xrk=Xrk, Xref=Xref, 
                #         Tend=Tend, tau=tau, torder=torder, rk=rk, 
                #         svals=svals, res_rk=res_rk, res_rf=res_rf)
            else:
                data = np.load(fname)
                X = data['X']
                res_rk = data['res_rk']
                res_rf = data['res_rf']  
                etime = tm()
            print(f' dt={tau:.0e}: time= {tm()-etime:5.2f}  res_RK: {rn(X,Xrk):.2e}, res_BS: {rn(X,Xref):.2e}, res: {CALE(X,L,Qc):.2e}')
            if torder == 1:
                axs[i,0].semilogy(np.linspace(0,Tend,len(res_rk)), res_rk, label=f'dt={tau:.2e}',marker='o')
                axs[i,1].semilogy(np.linspace(0,Tend,len(res_rf)), res_rf, label=f'dt={tau:.2e}',marker='o')
            else:
                axs[i,0].semilogy(np.linspace(0,Tend,len(res_rk)), res_rk, label=f'dt={tau:.2e}',marker='o',linestyle='--')
                axs[i,1].semilogy(np.linspace(0,Tend,len(res_rf)), res_rf, label=f'dt={tau:.2e}',marker='o',linestyle='--')
            erk[it, i, j] = res_rk[-1]
            erf[it, i, j] = res_rf[-1]
            nrm[it, i, j] = np.linalg.norm(X)
        #print('')

for i, rk in enumerate(rkv):
    for ax in axs[i,:]:
        ax.set_title(f'rk = {rk:02d}')
        ax.set_xlabel('dt')
        ax.set_ylim(1e-10,1)
        ax.legend()
axs[0,0].set_title(f'error vs. RK45\nrk = {rkv[0]:02d}')
axs[0,1].set_title(f'error vs. BS\nrk = {rkv[0]:02d}')
plt.xlabel('dt')
plt.legend()   

'''
fig, ax = plt.subplots(1,2)
ax[0].plot(np.linspace(0, Trk*Nrep+1, len(nrmX)), nrmX)
ax[1].semilogy(np.linspace(0, Trk*Nrep+1, len(nrmX)), erel[0,:])
'''
sys.exit()
fig1, ax1 = plt.subplots(1,2)
p = ax1[0].contourf(px, py, Xref, 100)
fig1.colorbar(p, ax = ax1[0])
ax1[0].set_title('Controllability')

p = ax1[1].contourf(px, py, Yref, 100)
fig1.colorbar(p, ax = ax1[1])
ax1[1].set_title('Observability')

fig2, ax2 = plt.subplots(1,1)
for i, rk in enumerate(rkv):
    ax2.loglog(tauv, erk[0, i, :], label=f'rk={rk:d}, order = 1',marker='o', color=f'C{i}')
    ax2.loglog(tauv, erk[1, i, :], label=f'rk={rk:d}, order = 2',marker='o', color=f'C{i}', linestyle='--')
ax2.legend()
plt.title('Error vs. RK45')
    
fig3, ax3 = plt.subplots(1,1)
for i, rk in enumerate(rkv):
    ax3.loglog(tauv, erf[0, i, :], label=f'rk={rk:d}, order = 1',marker='o', color=f'C{i}')
    ax3.loglog(tauv, erf[1, i, :], label=f'rk={rk:d}, order = 2',marker='o', color=f'C{i}', linestyle='--')
ax3.axhline(y = errRK, color='k', linestyle = '--')
ax3.legend()
plt.title('Error vs. BS')

fig4, ax4 = plt.subplots(1, len(rkv))
torder = 2
for i, rk in enumerate(rkv):
    print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
    tau = tauv[-3]
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_rk_{rk:02d}_dt{tau:.2e}_TO{torder:1d}.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    X = data['X']
    #Xr = X[:Nxc,:Nxc]
    #p = ax4[i].contourf(prx, pry, Xr, 100, cmap='RdBu_r')
    p = ax4[i].contourf(px, py, X, 100, cmap='RdBu_r')
    fig1.colorbar(p, ax = ax4[i])
    ax4[i].set_title(f'rk = {rk:2d}')

_, srk, _ = np.linalg.svd(Xrk)

fig5, ax5 = plt.subplots(1, len(rkv))
torder = 2
for i, rk in enumerate(rkv):
    print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
    tau = tauv[-3]
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_rk_{rk:02d}_dt{tau:.2e}_TO{torder:1d}.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    svals = data['svals']
    for s in srk:
        ax5[i].axhline(y=s, linestyle='--', color='k')
    for ir in range(rk):
        ax5[i].scatter(np.linspace(0,Tend,len(svals)), svals[:,ir], 40, 'r')
for ax in ax5:
    ax.set_yscale('log')
    
fig6, ax6 = plt.subplots(1, 4)
_, srk, _ = np.linalg.svd(Xrk)
_, srf, _ = np.linalg.svd(Xref)
nrmref = np.linalg.norm(Xref)
torder = 1
Tvrf = np.linspace(0,Tend,Nrep+1)
# reference
for i in range(0, Nx, 2):
    ax6[0].plot(Tvrf, nrmX, color='k')
    ax6[1].semilogy(Tvrf, abs(nrmX - nrmref), color='k')
    ax6[2].semilogy(Tvrf, sref[:,i], linestyle='--', color='k')
for i, rk in enumerate(rkv[::-1]):
    print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
    tau = tauv[-3]
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_rk_{rk:02d}_dt{tau:.2e}_TO{torder:1d}.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    svals = data['svals']
    Tvdlra = np.linspace(0,Tend,len(svals))
    nrmdlra = np.sqrt(np.sum(svals**2,1))
    ax6[0].plot(Tvdlra, nrmdlra, color=f'C{i}')
    ax6[1].semilogy(Tvdlra, abs(nrmdlra - nrmref), color=f'C{i}')
    for ir in range(rk):
        #ax6.scatter(np.linspace(0,Tend,len(svals)), abs(svals[:,ir]-srf[ir]), 30, f'C{i}')
        #ax6.semilogy(np.linspace(0,Tend,len(svals)), abs(svals[:,ir]-srf[ir]), color=f'C{i}')
        ax6[2].semilogy(Tvdlra, svals[:,ir], color=f'C{i}')
        ax6[3].semilogy(Tvdlra, abs(svals[:,ir]-srf[ir]), color=f'C{i}')

#ax6[1].set_yscale('log')
#ax6[1].set_ylim([1e-10, 1e3])

fig7, ax7 = plt.subplots(1,1)
torder = 2
for i, rk in enumerate(rkv):
    print(f'LR OSI: rk = {rk:2d}, torder={torder:1d}')
    tau = tauv[-3]
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_rk_{rk:02d}_dt{tau:.2e}_TO{torder:1d}.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    X = data['X']
    _,S,_ = np.linalg.svd(X)
    ax7.scatter(range(Nx), S, 60, f'C{i}', label=f'{rk:2d}')
ax7.scatter(range(Nx), srk, 60, 'k', marker='+')
ax7.set_yscale('log')
ax7.set_ylim([1e-12, 1e3])
ax7.set_xlim([0, 60])
ax7.legend()