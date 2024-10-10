#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os, sys
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from numpy.linalg import norm

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.H + Q
    return dXdt.flatten()
def Xdotr(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

plt.close("all")

base = '/home/skern/projects/LightROM/local/'

Nx = 128

# Parameters
L  = 50
x0 = -L/2                      # beginning of spatial domain
x1 = L/2                       # end of spatial domain
xv = np.linspace(x0, x1, Nx+2)

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.2
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__   = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xv,mu0,cu,cd,U,mu2,True)
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
DM1f,DM1b,DM2c = FDmat(xv)

# integration weights
dx = np.diff(xv)
#w = np.zeros(Nx+2)
#w[:Nx+1] += 0.5*dx
#w[1:]    += 0.5*dx
w = np.ones(Nx+2)*dx[0]

# linear operator
L = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

Nxc = Nx
Nxr = 2*Nx

# Complex case
Lc = np.matrix(L[1:-1,1:-1])
wc = w[1:-1]
xc = xv[1:-1]
# weight matrix for convenience
Wc = np.diag(wc)
# Input & output
Bc = np.zeros((Nxc, rkb))
Bc[:,0] = np.exp(-((xc - x_b)/s_b)**2)
Cc = np.zeros((rkc, Nxc))
Cc[0,:] = np.exp(-((xc - x_c)/s_c)**2)

# Real case
Lrp = np.real(L[1:-1,1:-1])
Lip = np.imag(L[1:-1,1:-1])
Lr  = np.block([[Lrp, -Lip],[Lip, Lrp]])
wr  = np.hstack(( w[1:-1], w[1:-1]))
xr  = np.hstack((xv[1:-1],xv[1:-1]))
xr  = xv[1:-1]
# weight matrix for convenience
Wr = np.diag(wr)
# Input & output
Brp = 0.5*np.real(Bc)
Bip = 0.5*np.imag(Bc)
Br  = np.block([[Brp, -Bip],[Bip, Brp]])
Crp = 0.5*np.real(Cc)
Cip = 0.5*np.imag(Cc)
Cr  = np.block([[Crp, -Cip],[Cip, Crp]])

# plotting
pxr, pyr = np.meshgrid(range(Nxr), range(Nxr))
pxc, pyc = np.meshgrid(xc,xc)
box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

#############################################
#
#      CONTROLLABILIITY
#
#############################################

# direct
Qbc = Bc @ Bc.T @ Wc
Xc = linalg.solve_continuous_lyapunov(Lc, -Qbc)

# direct
Qbr = Br @ Br.T @ Wr
Xr = linalg.solve_continuous_lyapunov(Lr, -Qbr)
#np.save(base + "data_BS_X.npy", Xr)
sys.exit()
"""
# FORCING DISTRIBUTION
fig, axs = plt.subplots(1,3)
p = axs[0].contourf(pxc, pyc, np.real(Qbc), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Re(Qbc)')
p = axs[1].contourf(pxc, pyc, np.imag(Qbc), 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Im(Qbc)')
p = axs[2].contourf(pxr, pyr, Qbr, 100)
fig.colorbar(p, ax = axs[2])
axs[2].set_title('Qbr')

for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])
    
# Complex Stuart-Bartels
fig, axs = plt.subplots(1,2)
p = axs[0].contourf(pxc, pyc, np.real(Xc), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Re(Xc)')
p = axs[1].contourf(pxc, pyc, np.imag(Xc), 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Im(Xc)')
for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

# Real Stuart-Bartels
fig, ax = plt.subplots(1,1)
p = ax.contourf(pxr, pyr, Xr, 100)
fig.colorbar(p, ax = ax)
ax.set_title('Xr')
ax.set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_ylim(ax.get_ylim()[::-1])

# Comparison
fig, axs = plt.subplots(1,4)
p = axs[0].contourf(pxc, pyc, Xr[Nx:,:Nx] - Xr[:Nx,Nx:], 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Xr[N:,:N] - Xr[:N,N:]')
p = axs[1].contourf(pxc, pyc, Xr[Nx:,:Nx] + Xr[:Nx,Nx:].T, 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Xr[N:,:N] + Xr[:N,N:].T')
p = axs[2].contourf(pxc, pyc, Xr[:Nx,:Nx] - Xr[Nx:,Nx:], 100)
fig.colorbar(p, ax = axs[2])
axs[2].set_title('Xr[:N,:N] - Xr[N:,N:]')
p = axs[3].contourf(pxc, pyc, Xr[:Nx,:Nx] + Xr[Nx:,Nx:], 100)
fig.colorbar(p, ax = axs[3])
axs[3].set_title('Xr[:N,:N] + Xr[N:,N:]')
for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

#############################################
#
#  OBSERVABILITY
#
#############################################
# direct
Qcc = Cc.T @ Cc @ Wc
Yc = linalg.solve_continuous_lyapunov(Lc.H, -Qcc)

Qcr = Cr.T @ Cr @ Wr
Yr = linalg.solve_continuous_lyapunov(Lr.T, -Qcr)
  
# FORCING DISTRIBUTION
fig, axs = plt.subplots(1,3)
p = axs[0].contourf(pxc, pyc, np.real(Qcc), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Re(Qcc)')
p = axs[1].contourf(pxc, pyc, np.imag(Qcc), 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Im(Qcc)')
p = axs[2].contourf(pxr, pyr, Qcr, 100)
fig.colorbar(p, ax = axs[2])
axs[2].set_title('Qcr')
for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])
    
# Complex Stuart-Bartels
fig, axs = plt.subplots(1,2)
p = axs[0].contourf(pxc, pyc, np.real(Yc), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Re(Yc)')
p = axs[1].contourf(pxc, pyc, np.imag(Yc), 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Im(Yc)')
for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

# Real Stuart-Bartels
fig, ax = plt.subplots(1,1)
p = ax.contourf(pxr, pyr, Yr, 100)
fig.colorbar(p, ax = ax)
ax.set_title('Yr')
ax.set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_ylim(ax.get_ylim()[::-1])

# Comparison
fig, axs = plt.subplots(1,4)
p = axs[0].contourf(pxc, pyc, Yr[Nx:,:Nx] - Yr[:Nx,Nx:], 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('Yr[N:,:N] - Yr[:N,N:]')
p = axs[1].contourf(pxc, pyc, Yr[Nx:,:Nx] + Yr[:Nx,Nx:].T, 100)
fig.colorbar(p, ax = axs[1])
axs[1].set_title('Yr[N:,:N] + Xr[:N,N:].T')
p = axs[2].contourf(pxc, pyc, Yr[:Nx,:Nx] - Yr[Nx:,Nx:], 100)
fig.colorbar(p, ax = axs[2])
axs[2].set_title('Yr[:N,:N] - Yr[N:,N:]')
p = axs[3].contourf(pxc, pyc, Yr[:Nx,:Nx] + Yr[Nx:,Nx:], 100)
fig.colorbar(p, ax = axs[3])
axs[3].set_title('Yr[:N,:N] + Yr[N:,N:]')
for ax in axs:
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])
"""
BBTWRKfort = np.load(base + f'data_GL_lyapconv_BBTW_RK_n{Nxc:04d}.npy')
BBTWRKc    = BBTWRKfort[:Nx, :Nx] + 1j*BBTWRKfort[Nx:, :Nx]

BBTWfort = np.load(base + f'data_GL_lyapconv_BBTW_DLRA_n{Nxc:04d}.npy')
BBTWc    = BBTWfort[:Nx, :Nx] + 1j*BBTWfort[Nx:, :Nx]

X0r      = np.load(base + f'data_GL_lyapconv_X0_RK_n{Nxc:04d}.npy')
X0c      = X0r[:Nx, :Nx] + 1j*X0r[Nx:, :Nx]

X1r      = np.load(base + f'data_GL_lyapconv_X_RK_n{Nxc:04d}_r{1:03d}.npy')
X1c      = X1r[:Nx, :Nx] + 1j*X1r[Nx:, :Nx]

# RK
tol = 1e-12
tspan = (0, 1)
nrep = 60
Xrkvr  = np.zeros((Nxr, Nxr, nrep+1))

#
'''
Xrkfile = 'Xrk_W.npy'
if not os.path.isfile(Xrkfile):
    print('Compute XRK (python):')
    X0 = X0r
    for i in range(nrep):
        etime = tm()
        sol = solve_ivp(Xdotr,tspan,X0.flatten(),args=(Lr,Qbr), atol=tol, rtol=tol)
        print(f'  RK step {i+1}: etime = {tm()-etime}')
        X0 = sol.y[:,-1].reshape(Lr.shape)
        Xrkvr[:, :, i+1] = X0.copy()
    np.save(Xrkfile, Xrkvr)
else:
    print('Load X_RK (python) '+ Xrkfile)
    Xrkvr = np.load(Xrkfile)

# Forcing
fig, axs = plt.subplots(3,3)
p = axs[0,0].contourf(pxr, pyr, Qbr, 100)
fig.colorbar(p, ax = axs[0,0])
axs[0,0].set_title('BBTWr python')
p = axs[0,1].contourf(pxr, pyr, BBTWfort, 100)
fig.colorbar(p, ax = axs[0,1])
axs[0,1].set_title('BBTWr fort')
p = axs[0,2].contourf(pxr, pyr, BBTWfort - Qbr, 100)
fig.colorbar(p, ax = axs[0,2])
axs[0,2].set_title('diff')
# Initial condition
p = axs[1,0].contourf(pxc, pyc, X0r[:Nx,:Nx], 100)
fig.colorbar(p, ax = axs[1,0])
axs[1,0].set_title('Re(X0)')
p = axs[1,1].contourf(pxc, pyc, X0r[Nx:,:Nx], 100)
fig.colorbar(p, ax = axs[1,1])
axs[1,1].set_title('Im(X0)')
p = axs[1,2].contourf(pxr, pyr, X0r, 100)
fig.colorbar(p, ax = axs[1,2])
axs[1,2].set_title('X0r')
# RK output
p = axs[2,0].contourf(pxr, pyr, Xrkvr[:,:,1], 100)
fig.colorbar(p, ax = axs[2,0])
axs[2,0].set_title('X_rk python')
p = axs[2,1].contourf(pxr, pyr, X1r, 100)
fig.colorbar(p, ax = axs[2,1])
axs[2,1].set_title('X_rk fortran')
p = axs[2,2].contourf(pxr, pyr, Xrkvr[:,:,1] - X1r, 100)
fig.colorbar(p, ax = axs[2,2])
axs[2,2].set_title('diff')
for ax in axs.ravel():
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_ylim(ax.get_ylim()[::-1])

XRKfort = [ X0r ]

for r in range(nrep):
    XDLRA.append(np.load(base + f'data_GL_X_RK_n{Nxc:04d}_r{r+1:03d}.npy'))
'''
'''
# RK EVOLUTION PY vs FORTRAN
fig, axs = plt.subplots(5,3)
for i in range(5):
    p = axs[i,0].contourf(pxr, pyr, Xrkvr[:,:,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,0])
    axs[i,0].set_title(f'PY t = {10*i+1}')
    p = axs[i,1].contourf(pxr, pyr, XRKfort[10*i+1], 100)
    fig.colorbar(p, ax = axs[i,1])
    axs[i,1].set_title(f'F t = {10*i+1}')
    p = axs[i,2].contourf(pxr, pyr, XRKfort[10*i+1] - Xrkvr[:,:,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,2])
    axs[i,2].set_title('diff')
    
for ax in axs.ravel():
    ax.axis('equal') 
    ax.axis('off')
    ax.set_ylim(ax.get_ylim()[::-1])
    
fig, axs = plt.subplots(5,3)
for i in range(5):
    p = axs[i,0].contourf(pxc, pyc, Xrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,0])
    axs[i,0].set_title(f'PY t = {10*i+1}')
    p = axs[i,1].contourf(pxc, pyc, XRKfort[10*i+1][:Nx,:Nx], 100)
    fig.colorbar(p, ax = axs[i,1])
    axs[i,1].set_title(f'F t = {10*i+1}')
    p = axs[i,2].contourf(pxc, pyc, XRKfort[10*i+1][:Nx,:Nx] - Xrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,2])
    axs[i,2].set_title('diff')
    
for ax, xd in zip(axs.ravel(), np.repeat(x_b, 15)):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
    ax.set_ylim(ax.get_ylim()[::-1])
'''
XUfiles = []
XSfiles = []
lbl = []

rkv = [ 6, 14, 40, 128, 256 ]
nrk = len(rkv)
TOv = np.repeat([1]    , nrk)
Nv  = np.repeat([ 128 ], nrk)
dtv = np.repeat([ '0.10E-04'], nrk)

for i, (N, torder, rk, dt) in enumerate(zip(Nv, TOv, rkv, dtv)):
    fnameU = f"data_GL_lyapconv_XU_n{N:04d}_TO{torder:d}_rk{rk:03d}_t{dt}.npy"
    XUfiles.append(base + fnameU)
    fnameS = f"data_GL_lyapconv_XS_n{N:04d}_TO{torder:d}_rk{rk:03d}_t{dt}.npy"
    XSfiles.append(base + fnameS)
    lbl.append(f'rk = {rk:d}, dt={dt}, TO={torder}')
   
file = f'data_GL_lyapconv_X_RK_n{Nxc:04d}_r{1:03d}.npy'
XRK_lyapconv_fort_ref = np.load(base + file)
print('Load '+file+' (comparison)')
    
XDLRA_lyapconv = [ X0r ]
# load DRLA data
for (XUf, XSf) in zip(XUfiles, XSfiles):
        print('Load '+XUf)
        XU = np.load(XUf)
        print('Load '+XSf)
        XS = np.load(XSf)
        XDLRA_lyapconv.append(XU @ XS @ XU.T @ Wr)

fig, axs = plt.subplots(nrk+1,2)
p = axs[0,0].contourf(pxr, pyr, BBTWfort, 100)
fig.colorbar(p, ax = axs[0,0])
axs[0,0].set_title('BBTWr')
p = axs[0,1].contourf(pxr, pyr, XRK_lyapconv_fort_ref, 100)
fig.colorbar(p, ax = axs[0,1])
axs[0,1].set_title('RK fort')

for i in range(nrk):
    p = axs[i+1,0].contourf(pxr, pyr, XDLRA_lyapconv[i], 100)
    fig.colorbar(p, ax = axs[i+1,0])
    axs[i+1,0].set_title(lbl[i])
    p = axs[i+1,1].contourf(pxr, pyr, XRK_lyapconv_fort_ref-XDLRA_lyapconv[i], 100)
    fig.colorbar(p, ax = axs[i+1,1])
    axs[i+1,1].set_title("diff to RK")
    
for ax in axs.ravel():
    ax.axis('equal') 
    ax.axis('off')
    ax.set_ylim(ax.get_ylim()[::-1])
    
sys.exit()
##########################################
#
# DLRA TEST
#
##########################################

XUfiles = []
XSfiles = []
lbl = []

rkv = [ 12, 12, 12, 12, 12, 12 ]
TOv = [1, 1, 1, 2, 2, 2]
Nv  = np.repeat([ 128 ], 6)
dtv = np.repeat([ '0.10E+00', '0.10E-01' ], 3)

for i, (N, torder, rk, dt) in enumerate(zip(Nv, TOv, rkv, dtv)):
    fnameU = f"data_GLXY_XU_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    print('Load '+fnameU)
    XUfiles.append(base + fnameU)
    fnameS = f"data_GLXY_XS_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    print('Load '+fnameS)
    XSfiles.append(base + fnameS)
    lbl.append(f'rk = {rk:d}, dt={dt}, TO={torder}')

fnameU = f'RK_test/data_GL_X_RK_n{Nxc:04d}_r{60:03d}.npy'
print('Load '+fnameU + ' (comparison)')
XRKfort_ref = np.load(base + fnameU)

XDLRA = []
# load DRLA data
for (XUf, XSf) in zip(XUfiles, XSfiles):
        XU = np.load(XUf)
        XS = np.load(XSf)
        XDLRA.append(XU @ XS @ XU.T @ Wr)
        
fig, axs = plt.subplots(nrk,2)
for i in range(nrk):
    p = axs[i,0].contourf(pxr, pyr, XDLRA[i], 100)
    fig.colorbar(p, ax = axs[i,0])
    axs[i,0].set_title(lbl[i])
    p = axs[i,1].contourf(pxr, pyr, XDLRA[i] - XRKfort_ref, 100)
    fig.colorbar(p, ax = axs[i,1])
    axs[i,1].set_title('diff (RK fort ref)')

for ax in axs.ravel():
    ax.axis('equal') 
    ax.axis('off')
    ax.set_ylim(ax.get_ylim()[::-1]) 