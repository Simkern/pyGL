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

X_files = []
N = 128

for irep in range(1):
    fnameU = f"data_GL_X_RK_n{N:04d}_r{irep+1:03d}.npy"
    X_files.append(base + fnameU)

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
wc = np.ones((Nxc+2,))

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
Xnr = np.linalg.norm(np.real(X))
Xni = np.linalg.norm(np.imag(X))
Xn = np.linalg.norm(X)

'''
BBTWfort = np.load(base + 'data_GL_BBTW_RK_n0128.npy')
BBTWc    = BBTWfort[:Nx, :Nx] + 1j*BBTWfort[Nx:, :Nx]

fig, axs = plt.subplots(2,2)

p = axs[0,0].contourf(px, py, np.real(Qb), 100)
fig.colorbar(p, ax = axs[0,0])
p = axs[0,1].contourf(px, py, np.imag(Qb), 100)
fig.colorbar(p, ax = axs[0,1])
p = axs[1,0].contourf(px, py, np.real(BBTWc), 100)
fig.colorbar(p, ax = axs[1,0])
p = axs[1,1].contourf(px, py, np.imag(BBTWc), 100)
fig.colorbar(p, ax = axs[1,1])
'''

# RK
tol = 1e-12
tspan = (0, 1)
nrep = 60
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
Xrnrm = [ np.linalg.norm(Xrkvr[:Nx,:Nx,i]) for i in range(nrep+1) ]
Xinrm = [ np.linalg.norm(Xrkvr[Nx:2*Nx,:Nx,i]) for i in range(nrep+1) ]
Xnrm = [ np.linalg.norm(Xrkvr[:,:,i]) for i in range(nrep+1) ]

# compute observability gramian

# direct
Qc = C.T @ C @ W
Y = linalg.solve_continuous_lyapunov(L.H, -Qc)
Ynr = np.linalg.norm(np.real(Y))
Yni = np.linalg.norm(np.imag(Y))
Yn = np.linalg.norm(Y)

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
Yrnrm = [ np.linalg.norm(Yrkvr[:Nx,:Nx,i]) for i in range(nrep+1) ]
Yinrm = [ np.linalg.norm(Yrkvr[Nx:2*Nx,:Nx,i]) for i in range(nrep+1) ]
Ynrm = [ np.linalg.norm(Yrkvr[:,:,i]) for i in range(nrep+1) ]

# norm
fig, axs = plt.subplots(1,2)
axs[0].semilogy(abs(Xrnrm - Xnr), label='RK - real', color='r')
axs[0].semilogy(abs(Xinrm - Xni), label='RK - imag', color='b')

axs[1].semilogy(abs(Yrnrm - Ynr), label='RK - real', color='r')
axs[1].semilogy(abs(Yinrm - Yni), label='RK - imag', color='b')
axs[1].legend()

Xfort = []
Xc    = []

for i, ifile in enumerate(X_files):
    Xfort.append(np.load(ifile))
    Xc.append(Xfort[-1][:Nx, :Nx] + 1j*Xfort[-1][Nx:, :Nx])

print('Check Lyapunov equation residuals ...')
for i in range(len(Xc)):
    print('%10s: %e' % (str(i+1), abs(L @ Xc[i] + Xc[i] @ L.H + B @ B.T @ W).max()))

fig, axs = plt.subplots(1,1)

p = axs.contourf(px, py, Xfort[-1][:Nx,:Nx], 100)
fig.colorbar(p, ax = axs)
    
fig, axs = plt.subplots(1,2)

p = axs[0].contourf(px, py, np.real(Qb), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title('B @ B.T @ W')
p = axs[1].contourf(px, py, np.real(Qc), 100)
fig.colorbar(p, ax = axs[1]) 
axs[1].set_title('C.T @ C @ W')

Xs = [ np.real(X), Xrkvr[:Nx,:Nx,-1] ]
Ys = [ np.real(Y), Yrkvr[:Nx,:Nx,-1] ]
labels = ['Bartels-Stuart','RK']
symbs = ['o','x']

# Spectra
print('Compute Spectra ... ', end=''); etime = tm()

cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, axs = plt.subplots(1,2)
for i, (Xi, Yi, lbl, s) in enumerate(zip(Xs, Ys, labels, symbs)):
    Sx = np.linalg.svd(np.real(Xi), full_matrices=False, compute_uv=False)
    axs[0].semilogy(Sx, color=cc[i])
    Sy = np.linalg.svd(np.real(Yi), full_matrices=False, compute_uv=False)
    axs[1].semilogy(Sy, label=lbl, color=cc[i])
    if not i==0:
        S = np.linalg.svd(np.real(X), full_matrices=False, compute_uv=False)
        axs[0].semilogy(abs(Sx-S), color=cc[i])
        S = np.linalg.svd(np.real(Y), full_matrices=False, compute_uv=False)
        axs[1].semilogy(abs(Sy-S), color=cc[i])    
for ax in axs.flatten():
    ax.set_xlim(0,70)
    ax.set_ylim(1e-14, 1e3)
axs[0].set_title('Controllability')
axs[1].set_title('Observability')
axs[0].set_title('Absolute difference to B-S')
axs[1].set_title('Absolute difference to B-S')
plt.suptitle('Singular values')
axs[1].legend()
print(f'done. etime = {tm()-etime}')

# Balanced truncation
Tv = []
Tinvv = []
Sv = []

print('Compute Balancing Transformations ... ', end=''); etime = tm()
fig, axs = plt.subplots(2,1)
for i, (Xi, Yi, lbl, s) in enumerate(zip(Xs, Ys, labels, symbs)):
    # SVD
    Up, Sp, _ = np.linalg.svd(Xi)
    Uq, Sq, _ = np.linalg.svd(Yi)
    # Cholesky
    Pc = Up @ np.diag(np.sqrt(Sp))
    Qc = Uq @ np.diag(np.sqrt(Sq))
    # BT
    Usvd, S, Vsvd = np.linalg.svd(Pc.T @ W @ Qc, full_matrices=False)
    sqrtSinv = np.diag(np.sqrt(1.0/S))
    Tv.append(sqrtSinv @ Vsvd @ Qc.T)
    Tinvv.append(Pc @ Usvd @ sqrtSinv)
    Sv.append(S)
    axs[0].semilogy(S, label=lbl, color=cc[i]) 
    if not i==0:
        axs[1].semilogy(abs(S - Sv[0]), color=cc[i])
for ax in axs:
    ax.set_xlim(0,70)
    ax.set_ylim(1e-14, 1e3)
axs[0].set_title('Hankel singular values')
axs[0].legend()
print(f'done. etime = {tm()-etime}')

# project system
print('Project ...', end=''); etime = tm()
systems = []
for T, Tinv in zip(Tv, Tinvv):
    systems.append((T @ W @ L @ Tinv, T @ W @ B, C @ Tinv @ W ))
print(f'done. etime = {tm()-etime}')

fig, axs = plt.subplots(2,2)

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
    
fig, axs = plt.subplots(1,2)

p = axs[0].contourf(px, py, Xrkvr[:Nx,:Nx,-1] - np.real(X), 100)
fig.colorbar(p, ax = axs[0])
axs[0].set_title(f'X RK (t={nrep})')
p = axs[1].contourf(px, py, Yrkvr[:Nx,:Nx,-1] - np.real(Y), 100)
fig.colorbar(p, ax = axs[1]) 
axs[1].set_title(f'Y RK (t={nrep})')

for ax, xd in zip(axs.ravel(), np.repeat(dot, 2, axis = 0).ravel()):
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'k')
    ax.scatter(xd, xd, 50, 'r')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('off')
plt.suptitle('Difference to direct solution')