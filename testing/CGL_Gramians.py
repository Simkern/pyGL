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

# naive Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.H + Q
    return dXdt.flatten()

plt.close("all")

make_real = False

#YDLRAfile = '/home/skern/projects/LightROM/local/run1_128/GL_Ydata_TO2_rk14_t3.npy'
#XDLRAfile = '/home/skern/projects/LightROM/local/run1_128/GL_Xdata_TO2_rk14_t3.npy'
YDLRAfile = '/home/skern/projects/LightROM/local/run2_128_lowdt/GL_Ydata_TO2_rk14_t3.npy'
XDLRAfile = '/home/skern/projects/LightROM/local/run2_128_lowdt/GL_Xdata_TO2_rk14_t3.npy'
YDLRAfile2 = '/home/skern/projects/LightROM/local/run3_128_rk20/GL_Ydata_TO2_rk20_t3.npy'
XDLRAfile2 = '/home/skern/projects/LightROM/local/run3_128_rk20/GL_Xdata_TO2_rk20_t3.npy'
YDLRAfile3 = '/home/skern/projects/LightROM/local/run4_128_rk40/GL_Ydata_TO2_rk40_t1.npy'
XDLRAfile3 = '/home/skern/projects/LightROM/local/run4_128_rk40/GL_Xdata_TO2_rk40_t1.npy'
YDLRAfile4 = '/home/skern/projects/LightROM/local/run4_128_rk40/GL_Ydata_TO2_rk40_t2.npy'
XDLRAfile4 = '/home/skern/projects/LightROM/local/run4_128_rk40/GL_Xdata_TO2_rk40_t2.npy'


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
DM1f,DM1b,DM2c = FDmat(xc)

# integration weights
dx = np.diff(xc)
wc = np.zeros(Nxc+2)
wc[:Nxc+1] += dx
wc[1:]     += dx

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

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

# plot input and output
fig = plt.figure(1)
plt.plot(B[:,0])
plt.plot(C[0,:])

# compute controllability gramian

# direct
Q = B @ B. T @ W
X = linalg.solve_continuous_lyapunov(L, -Q)

# RK
tol = 1e-12
tspan = (0, 1)
nrep = 50
Xrkv  = np.zeros((Nx, Nx, nrep+1), dtype=complex)
M_out = np.zeros((2*Nx, 2*Nx, nrep+1))
#
XRKfile = 'Xrk.npy'
if not os.path.isfile(XRKfile):
    print('Compute X:')
    U0 = np.random.random_sample((Nx, rkX0)) + 1j*np.random.random_sample((Nx, rkX0))
    X0 = U0 @ U0.T
    Xr = np.real(X0)
    Xi = np.imag(X0)
    M_out[:, :, 0] = np.block([[Xr, -Xi],[Xi, Xr]])
    for i in range(nrep):
        etime = tm()
        sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(L,Q), atol=tol, rtol=tol)
        print(f'  RK step {i+1}: etime = {tm()-etime}')
        X0 = sol.y[:,-1].reshape(L.shape)
        Xr = np.real(X0)
        Xi = np.imag(X0)
        M_out[:, :, i+1] = np.block([[Xr, -Xi],[Xi, Xr]])
    np.save(XRKfile, M_out)
    Xrkvr = M_out.copy()
else:
    Xrkvr = np.load(XRKfile)

# load DRLA data
XDLRA = np.load(XDLRAfile)

# compute observability gramian

# direct
Q = C.T @ C @ W
Y = linalg.solve_continuous_lyapunov(L.H, -Q)

# RK
YRKfile = 'Yrk.npy'
if not os.path.isfile(YRKfile):
    print('Compute Y:')
    U0 = np.random.random_sample((Nx, rkX0)) + 1j*np.random.random_sample((Nx, rkX0))
    X0 = U0 @ U0.T
    Xr = np.real(X0)
    Xi = np.imag(X0)
    M_out[:, :, 0] = np.block([[Xr, -Xi],[Xi, Xr]])
    for i in range(nrep):
        etime = tm()
        sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(L.H,Q), atol=tol, rtol=tol)
        print(f'  RK step {i+1}: etime = {tm()-etime}')
        X0 = sol.y[:,-1].reshape(L.shape)
        Xr = np.real(X0)
        Xi = np.imag(X0)
        M_out[:, :, i+1] = np.block([[Xr, -Xi],[Xi, Xr]])
    np.save(YRKfile, M_out)
    Yrkvr = M_out.copy()
else:
    Yrkvr = np.load(YRKfile)
    
# load DRLA data
YDLRA = np.load(YDLRAfile)

# rk20 data
# load DRLA data
XDLRA2 = np.load(XDLRAfile2)
YDLRA2 = np.load(YDLRAfile2)
XDLRA3 = np.load(XDLRAfile3)
YDLRA3 = np.load(YDLRAfile3)
XDLRA4 = np.load(XDLRAfile4)
YDLRA4 = np.load(YDLRAfile4)

Xs = (np.real(X), Xrkvr[:Nx,:Nx,-1],XDLRA[:Nx,:Nx],XDLRA2[:Nx,:Nx],XDLRA3[:Nx,:Nx],XDLRA4[:Nx,:Nx])
Ys = (np.real(Y), Yrkvr[:Nx,:Nx,-1],YDLRA[:Nx,:Nx],YDLRA2[:Nx,:Nx],YDLRA3[:Nx,:Nx],YDLRA4[:Nx,:Nx])

labels = ['Bartels-Stuart','RK','DLRA rk14','DLRA rk20', 'DLRA rk40 0.1', 'DLRA rk40 0.01']
symbs = ['o','x','d','*', 's', '^']

# Spectra
print('Compute Spectra ... ', end=''); etime = tm()
fig, axs = plt.subplots(1,2)
for X, Y, lbl, s in zip(Xs, Ys, labels, symbs):
    S = np.linalg.svd(np.real(X), full_matrices=False, compute_uv=False)
    axs[0].semilogy(S, s)
    S = np.linalg.svd(np.real(X), full_matrices=False, compute_uv=False)
    axs[1].semilogy(S, s, label=lbl)
for ax in axs:
    ax.set_xlim(0,70)
    ax.set_ylim(1e-14, 1e3)
axs[0].set_title('Controllability')
axs[1].set_title('Observability')
plt.suptitle('Singular values')
axs[1].legend()
print(f'done. etime = {tm()-etime}')

# Balanced truncation
Tv = []
Tinvv = []
Sv = []

print('Compute Balancing Transformations ... ', end=''); etime = tm()
fig = plt.figure()
for X, Y, lbl, s in zip(Xs, Ys, labels, symbs):
    # SVD
    Up, Sp, _ = np.linalg.svd(X)
    Uq, Sq, _ = np.linalg.svd(Y)
    # Cholesky
    Pc = Up @ np.diag(np.sqrt(Sp))
    Qc = Uq @ np.diag(np.sqrt(Sq))
    # BT
    Usvd, S, Vsvd = np.linalg.svd(Pc.T @ Qc, full_matrices=False)
    sqrtSinv = np.diag(np.sqrt(1.0/S))
    Tv.append(sqrtSinv @ Vsvd @ Qc.T)
    Tinvv.append(Pc @ Usvd @ sqrtSinv)
    Sv.append(S)
    plt.semilogy(S, s, label=lbl) 
plt.xlim(0,70)
plt.ylim(1e-14, 1e3)
plt.title('Hankel singular values')
plt.legend()
print(f'done. etime = {tm()-etime}')

# project system
print('Project ...', end=''); etime = tm()
systems = []
for T, Tinv in zip(Tv, Tinvv):
    systems.append((T @ L @ Tinv, T @ B, C @ Tinv))
print(f'done. etime = {tm()-etime}')

print('Compute Differential Bode plots ... ', end=''); etime = tm()
# Bode
nf = 20
f = np.linspace(-2,2,nf)
Gfull = np.zeros(nf)
fig = plt.figure()
I = np.eye(L.shape[0])
for i, g in enumerate(f):
    Gg = C @ np.linalg.inv(1j*g*I - L) @ B
    Sg = np.linalg.svd(Gg, full_matrices=False, compute_uv=False)
    Gfull[i] = Sg[0]
G = np.zeros(nf)
for system, lbl in zip(systems, labels):
    Ah, Bh, Ch = system
    I = np.eye(Ah.shape[0])
    for i, g in enumerate(f):
        Gg = Ch @ np.linalg.inv(1j*g*I - Ah) @ Bh
        Sg = np.linalg.svd(Gg, full_matrices=False, compute_uv=False)
        G[i] = Sg[0]
    plt.semilogy(f, abs(G-Gfull), label=lbl)
plt.title('Transfer functions')
plt.legend()
print(f'done. etime = {tm()-etime}')

# plotting
px, py = np.meshgrid(x,x)
box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

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

p = axs[2,0].contourf(px, py, XDLRA[:Nx,:Nx]*(2/3), 100)
fig.colorbar(p, ax = axs[2,0])
axs[2,0].set_title(f'2/3*DLRA (t={nrep})')
p = axs[2,1].contourf(px, py, YDLRA[:Nx,:Nx]*(2/3), 100)
fig.colorbar(p, ax = axs[2,1]) 
axs[2,1].set_title(f'2/3*DLRA (t={nrep})')

for ax, xd in zip(axs.ravel(), np.repeat(dot, 3, axis = 0).ravel()):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')

fig, axs = plt.subplots(5,2)
for i in range(5):
    p = axs[i,0].contourf(px, py, Xrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,0])
    axs[i,0].set_title(f't = {10*i+1}')
    p = axs[i,1].contourf(px, py, Yrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,1])
    axs[i,1].set_title(f't = {10*i+1}')
    
for ax, xd in zip(axs.ravel(), np.repeat(dot, 5, axis = 0).ravel()):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
    
fig, axs = plt.subplots(2,2)

p = axs[0,0].contourf(px, py, Xrkvr[:Nx,:Nx,-1] - np.real(X), 100)
fig.colorbar(p, ax = axs[0,0])
axs[0,0].set_title(f'X RK (t={nrep})')
p = axs[0,1].contourf(px, py, Yrkvr[:Nx,:Nx,-1] - np.real(Y), 100)
fig.colorbar(p, ax = axs[0,1]) 
axs[0,1].set_title(f'X RK (t={nrep})')

p = axs[1,0].contourf(px, py, 2/3*XDLRA[:Nx,:Nx] - np.real(X), 100)
fig.colorbar(p, ax = axs[1,0])
axs[1,0].set_title(f'Y DLRA (t={nrep})')
p = axs[1,1].contourf(px, py, 2/3*YDLRA[:Nx,:Nx] - np.real(Y), 100)
fig.colorbar(p, ax = axs[1,1]) 
axs[1,1].set_title(f'Y DLRA (t={nrep})')

for ax, xd in zip(axs.ravel(), np.repeat(dot, 2, axis = 0).ravel()):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'k')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
plt.suptitle('Difference to direct solution')

'''
fig, axs = plt.subplots(5,2)
for i in range(5):
    p = axs[i,0].contourf(px, py, Xrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,0])
    axs[i,0].set_title(f't = {10*i+1}')
    p = axs[i,1].contourf(px, py, Yrkvr[:Nx,:Nx,10*i+1], 100)
    fig.colorbar(p, ax = axs[i,1])
    axs[i,1].set_title(f't = {10*i+1}')
for ax in axs.ravel():
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
'''