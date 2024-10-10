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
from core.utils import p as pmat

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.H + Q
    return dXdt.flatten()

plt.close("all")

make_real = False

base = '/home/skern/projects/LightROM/local/'
#base = '/home/skern/projects/LightROM/local/Xctl_Yobs_data/data_npy/'

XUfiles = []
XSfiles = []
YUfiles = []
YSfiles = []
lbls    = []

'''
TOv = [ 1, 1, 1, 1, 1, 1 ]
Nv  = [128, 128, 128, 128, 128, 128]
rkv = [ 2, 6, 10, 14, 20, 40 ]
dtv = [ '0.10E-02','0.10E-02','0.10E-02','0.10E-02','0.10E-02','0.10E-02' ]
'''
TOv = [ 1 ]
Nv  = [ 256 ]
rkv = [ 12 ]
dtv = [ '0.10E+00']

for i, (N, torder, rk, dt) in enumerate(zip(Nv, TOv, rkv, dtv)):
    fnameU = f"data_GL_XU_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    XUfiles.append(base + fnameU)
    fnameS = f"data_GL_XS_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    XSfiles.append(base + fnameS)
    fnameU = f"data_GL_YU_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    YUfiles.append(base + fnameU)
    fnameS = f"data_GL_YS_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
    YSfiles.append(base + fnameS)
    lbls.append('DLRA rk'+str(rk)+' '+dt)

# Parameters
L  = 50
x0 = -L/2                      # beginning of spatial domain
x1 = L/2                       # end of spatial domain
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
DM1f,DM1b,DM1c,DM2c = FDmat(xc)

# integration weights
dx = np.diff(xc)
wc = np.zeros(Nxc+2)
#wc[:Nxc+1] += 0.5*dx
#wc[1:]     += 0.5*dx
wc = np.ones((Nxc+2,))*dx[0]

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1c + gamma*DM2c)

if make_real:
    # make real
    Lr = np.real(Lc[1:-1,1:-1])
    Li = np.imag(Lc[1:-1,1:-1])
    L  = np.matrix([[Lr, -Li],[Li, Lr]])
    Nx = 2*Nxc
    w  = np.hstack((wc[1:-1],wc[1:-1]))
    x  = np.hstack((xc[1:-1],xc[1:-1]))
else:
    L = np.matrix(Lc[1:-1,1:-1])
    w = wc[1:-1]
    x = xc[1:-1]
    Nx = Nxc

# weight matrix for convenience
W = np.diag(w)

# Input & output
B = np.zeros((Nx, rkb))
B[:,0] = np.exp(-((x - x_b)/s_b)**2)
C = np.zeros((rkc, Nx))
C[0,:] = np.exp(-((x - x_c)/s_c)**2)

# plotting
px, py = np.meshgrid(x,x)
box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

# compute controllability gramian

# direct
Qb = B @ B. T @ W
X = linalg.solve_continuous_lyapunov(L, -Qb)

# RK
tol = 1e-12
tspan = (0, 1)
nrep = 60
Xrkv  = np.zeros((Nx, Nx, nrep+1), dtype=complex)
M_out = np.zeros((2*Nx, 2*Nx, nrep+1))
#
XRKfile = 'Xrk_W.npy'
if not os.path.isfile(XRKfile):
    print('Compute X:')
    U0 = np.random.random_sample((Nx, rkX0)) + 1j*np.random.random_sample((Nx, rkX0))
    X0 = U0 @ U0.T
    Xr = np.real(X0)
    Xi = np.imag(X0)
    M_out[:, :, 0] = np.block([[Xr, -Xi],[Xi, Xr]])
    for i in range(nrep):
        etime = tm()
        sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(L,Qb), atol=tol, rtol=tol)
        print(f'  RK step {i+1}: etime = {tm()-etime}')
        X0 = sol.y[:,-1].reshape(L.shape)
        Xr = np.real(X0)
        Xi = np.imag(X0)
        M_out[:, :, i+1] = np.block([[Xr, -Xi],[Xi, Xr]])
    np.save(XRKfile, M_out)
    Xrkvr = M_out.copy()
else:
    Xrkvr = np.load(XRKfile)
# Reconstruct complex solution
Xrk = Xrkvr[:Nx, :Nx, -1] + 1j*Xrkvr[Nx:, :Nx, -1]

# compute observability gramian

# direct
Qc = C.T @ C @ W
Y = linalg.solve_continuous_lyapunov(L.H, -Qc)

# RK
YRKfile = 'Yrk_W.npy'
if not os.path.isfile(YRKfile):
    print('Compute Y:')
    U0 = np.random.random_sample((Nx, rkX0)) + 1j*np.random.random_sample((Nx, rkX0))
    X0 = U0 @ U0.T
    Xr = np.real(X0)
    Xi = np.imag(X0)
    M_out[:, :, 0] = np.block([[Xr, -Xi],[Xi, Xr]])
    for i in range(nrep):
        etime = tm()
        sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(L.H,Qc), atol=tol, rtol=tol)
        print(f'  RK step {i+1}: etime = {tm()-etime}')
        X0 = sol.y[:,-1].reshape(L.shape)
        Xr = np.real(X0)
        Xi = np.imag(X0)
        M_out[:, :, i+1] = np.block([[Xr, -Xi],[Xi, Xr]])
    np.save(YRKfile, M_out)
    Yrkvr = M_out.copy()
else:
    Yrkvr = np.load(YRKfile)
# Reconstruct complex solution
Yrk = Yrkvr[:Nx, :Nx, -1] + 1j*Yrkvr[Nx:, :Nx, -1]

Xc = [ X, Xrk ]
Yc = [ Y, Yrk ]
labels = ['Bartels-Stuart','RK']
symbs = ['o','x']

# Read data from DLRA

XDLRA = []
YDLRA = []
XDLRAc = []
YDLRAc = []

Nxf = 256
    
# load DRLA data
for (XUf, XSf, YUf, YSf) in zip(XUfiles, XSfiles, YUfiles, YSfiles):
    XU = np.load(XUf)
    XS = np.load(XSf)
    YU = np.load(YUf)
    YS = np.load(YSf)
    XDLRA.append(XU @ XS @ XU.T)
    YDLRA.append(YU @ YS @ YU.T)
    # Reconstruct complex solution
    Xc.append(XDLRA[-1][:Nxf, :Nxf] + 1j*XDLRA[-1][Nxf:, :Nxf])
    Yc.append(YDLRA[-1][:Nxf, :Nxf] + 1j*YDLRA[-1][Nxf:, :Nxf])

Xs = np.real(Xc)
Ys = np.real(Yc)

labels += lbls
symbs  += ['d','*', 's', '^']

print('Check Lyapunov equation residuals ...')
for i in range(len(Xc)):
    print('%10s: %e' % (labels[i], abs(L @ Xc[i] + Xc[i] @ L.H + B @ B.T @ W).max()))

cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, axs = plt.subplots(3,2)

p = axs[0,0].contourf(px, py, np.real(X), 100)
fig.colorbar(p, ax = axs[0,0])
axs[0,0].set_title('Controllability')
p = axs[0,1].contourf(px, py, np.real(Y), 100)
fig.colorbar(p, ax = axs[0,1]) 
axs[0,1].set_title('Observability')

p = axs[1,0].contourf(px, py, Xrkvr[:Nx,:Nx,-1], 100)
fig.colorbar(p, ax = axs[1,0])
axs[1,0].set_title(f'RK (t={nrep})')
p = axs[1,1].contourf(px, py, Yrkvr[:Nx,:Nx,-1], 100)
fig.colorbar(p, ax = axs[1,1]) 
axs[1,1].set_title(f'RK (t={nrep})')

if len(XDLRA) > 0:
    p = axs[2,0].contourf(px, py, XDLRA[0][:Nx,:Nx], 100)
    fig.colorbar(p, ax = axs[2,0])
    axs[2,0].set_title(f'DLRA (t={nrep})')
    p = axs[2,1].contourf(px, py, YDLRA[0][:Nx,:Nx], 100)
    fig.colorbar(p, ax = axs[2,1]) 
    axs[2,1].set_title(f'DLRA (t={nrep})')

for ax, xd in zip(axs.ravel(), np.repeat(dot, 3, axis = 0).ravel()):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
