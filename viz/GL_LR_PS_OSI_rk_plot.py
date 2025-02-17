#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, copy
import numpy as np
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('../.')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_test

params = {'text.usetex': False,
          'font.size': 25,
          'axes.labelsize' : 16, 
          'xtick.labelsize': 12,  # X-tick label font size
          'ytick.labelsize': 12,  # Y-tick label font size
          'legend.fontsize': 14,
          'legend.handlelength': 1.,}
plt.rcParams.update(params)

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
_, srf, _ = np.linalg.svd(Xref)

# Load RK45 data
Tend = 50
filename = f'Xrk_CGL_Nx{Nx:02d}_rk0_{rk_X0:02d}.npz'
fname = os.path.join(fldr,filename)
data = np.load(fname)
Xrkv = data['Xrkv']
erel = data['rel_error_Xref']
Tv   = data['Tv']
nrmX = []
sref = np.zeros((len(Tv),Nx))
for i in range(len(Tv)):
    nrmX.append(np.linalg.norm(Xrkv[:,:,0,i]))
    _, sref[i,:], _ = np.linalg.svd(Xrkv[:,:,0,i])
Xrk = np.squeeze(Xrkv[:,:,0,Tv==Tend])
errRK = rn(Xrk, Xref)        
  
# compare RK45 to LR_OSI
TOv = [1, 2]
tspan = (0,Tend)
tauv = np.logspace(0, -3, 7)
tolrkv = [ 1e-2, 1e-6 ]
tol = 1e-12

erk = np.zeros((len(TOv), len(tauv), len(tolrkv)))
erf = np.zeros((len(TOv), len(tauv), len(tolrkv)))
nrm = np.zeros((len(TOv), len(tauv), len(tolrkv)))
for i, torder in enumerate(TOv):
    print(f'LR RA-OSI: torder={torder:1d}')
    icol = 0
    for j, tau in enumerate(tauv):
        for k, tolrk in enumerate(tolrkv): 
            print(f' dt={tau:.0e}:')
            filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_dt{tau:.2e}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive.npz'
            fname = os.path.join(fldr,filename)
            data = np.load(fname)
            X = data['X']
            res_rk = data['res_rk']
            res_rf = data['res_rf']  
            etime = tm()
            print(f' dt={tau:.0e}:  etime = {tm()-etime:5.2f}   rel error: {rn(X,Xrk):.4e}')
            erk[i, j, k] = res_rk[-1]
            erf[i, j, k] = res_rf[-1]
            nrm[i, j, k] = np.linalg.norm(X)
        print('')

# plotting

tau = tauv[-1]
torder = 2

# singular values
fig1, ax1 = plt.subplots(1,1)
for i, tolrk in enumerate(tolrkv[::-1]):
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_dt{tau:.2e}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    svals = data['svals']
    Tvdlra = np.linspace(0,Tend,len(svals))
    ax1.scatter(range(1, svals.shape[1]+1), svals[-1,:], 40, color=f'C{i}', label=f'$tol = {tolrk:.0e}$')
    plt.axhline(y = tolrk, color = f'C{i}', linestyle='--')
ax1.scatter(range(1,Nx+1), srf, 60, 'k', marker='+', label='exact')
ax1.set_yscale('log')
ax1.set_ylim(1e-8, 1e3)
ax1.set_xlim([0, 60])
# Reverse legend order
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])   
plt.xlabel('#')
plt.ylabel(r'$\sigma$')

# norm over integration time vs BS
fig2, ax2 = plt.subplots(1,1)

for i, tolrk in enumerate(tolrkv[::-1]):
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_dt{tau:.2e}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    res_rf = data['res_rf']
    Tvdlra = np.linspace(0,Tend,len(res_rf))
    ax2.semilogy(Tvdlra, res_rf, color=f'C{i}') 
ax2.set_ylim(1e-8, 1e0)
plt.xlabel('time')
plt.ylabel(r'$\| X - X_{ref} \|_2/N$')
# error norm vs BS

fig3, ax3 = plt.subplots(1,1)

for i, tolrk in enumerate(tolrkv[::-1]):
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_dt{tau:.2e}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    rkv = data['rkvec']
    Tvdlra = np.linspace(0,Tend,len(rkv))
    ax3.plot(Tvdlra, rkv, color=f'C{i}')
plt.xlabel('time')
plt.ylabel(r'$rank(X)$')

fig4, ax4 = plt.subplots(1,1)

for i, tolrk in enumerate(tolrkv[::-1]):
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_dt{tau:.2e}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive.npz'
    fname = os.path.join(fldr,filename)
    data = np.load(fname)
    res_rf = data['res_rf']
    Tvdlra = np.linspace(0,Tend,len(res_rf))
    ax4.semilogy(Tvdlra, res_rf, color=f'C{i}') 
    
for k, tolrk in enumerate(tolrkv[1:2]):
    filename = f'Xdlra_CGL_Nx{Nxc:02d}_rk0_{rk_X0:02d}_TO{torder:1d}_tolrk_{tolrk:.0e}_rankadaptive_var_dt.npz'
    fname = os.path.join(fldr,filename)
    Tvec = [ 20, 10, 10, 10 ]
    dtv  = [ 1.0, 0.1, 0.01, 0.001 ]
    ifinitv = [ True, False, False, False, False ]
    data = np.load(fname, allow_pickle=True)
    X = data['X']
    res_rk = data['res_rk']
    res_rf = data['res_rf']  
    tvec = data['tvec']  
    rkvec = data['rkvec']
    etime = data['etime']
    ax4.semilogy(tvec, res_rf, label=f'variable $dt$', linestyle='--', color='k')
ax4.set_ylim(1e-8, 1e0)
plt.xlabel('time')
plt.ylabel(r'$\| X - X_{ref} \|_2/N$')
plt.legend()


for fig in [ fig1, fig2, fig3, fig4 ]:
    fig.set_figheight(5)
    fig.set_figwidth(4)   

if if_save:
    names = ['spectrum_rkad',
             'err_nrm_vs_time_rkad',
             'rk_vs_time_rkad',
             'var_dt']
    if make_real:#
        for i in range(len(names)):
            names[i] += '_real'
    for fig, name in zip([fig1, fig2, fig3, fig4], names):
        fig.savefig(os.path.join('pics_png',name+'.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(os.path.join('pics_eps',name+'.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)