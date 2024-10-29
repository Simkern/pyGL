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
from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_rk_test
from solvers.lyapunov_ProjectorSplittingIntegrator import LR_OSI_rk_test_2

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
wc[:Nxc+1] = dx
wc[1:]     = dx

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
#np.save('CGL_Lyapunov_Controllability_Xref_BS_W.npy', Xref)

Qo = C.T @ C @ W
Yref = linalg.solve_continuous_lyapunov(L.T, -Qo)
#np.save('CGL_Lyapunov_Observability_Yref_BS_W.npy', Yref)

nQ    = np.linalg.norm(Qc)
nA    = np.linalg.norm(L)

filename = f'X0_CGL_Nx{Nx:02d}_rk_X0_{rk_X0:02d}_test.npz'
fname = os.path.join(fldr,filename)
if not os.path.isfile(fname):
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((Nx, rk_X0)),mode='economic')
    S0    = np.diag(sorted(s0)[::-1]);
    X0    = U0 @ S0 @ U0.T
    np.savez(fname, X0=X0, S0=S0, U0=U0)
    filename = f'CGL_Nx{Nx:02d}_U0_r_X0_{rk_X0:02d}.npy'
    np.save(os.path.join(fldr,filename), U0)
    filename = f'CGL_Nx{Nx:02d}_S0_r_X0_{rk_X0:02d}.npy'
    np.save(os.path.join(fldr,filename), S0)
else:
    data = np.load(fname)
    X0   = data['X0']
    S0   = data['S0']
    U0   = data['U0']
    
U0_svd,S0_svd,V0_svdh = linalg.svd(X0, full_matrices=False)

print(' '.join(f'{s:.2e}' for s in S0_svd[:12]))

x0 = np.ones(Nx).reshape(-1,1)
#print(B.T @ x0)


#p = Qc @ x0
#for i in range(Nx):
#    print(f'{i+1}: {p[i]}')

#print(f'maxval: {p.max()}')
#x0 = np.zeros(Nx).reshape(-1,1)
#x0[0] = 1.0
#print(linalg.expm(0.1*L) @ x0)



#sys.exit()

# compare RK45 to LR_OSI
TOv = [1]
Tend = 5
tspan = (0,Tend)
tauv = [ 0.1 ]
tolrkv = [ 1e-2 ] #, 1e-6, 1e-10 ]
tol = 1e-12

for i, torder in enumerate(TOv):
    print(f'LR RA-OSI: torder={torder:1d}')
    icol = 0
    for j, tau in enumerate(tauv):
        for k, tolrk in enumerate(tolrkv): 
            print(f' dt={tau:.0e}:')
            etime = tm()
            nt = int(np.round(Tend/tau))
            X00 = copy.deepcopy(X0)
            etime = tm()
            U, S, svals, res_rf, rkvec, tvec = LR_OSI_rk_test_2(L, Qc, X00, Xref, Tend, tau, torder=torder, verb=1, tol=tolrk)
            etime = tm() - etime
            X = U @ S @ U.T
            #np.savez(fname, X=X, U=U, S=S, X0=X0, Qc=Qc, Xrk=Xrk, Xref=Xref, 
            #         Tend=Tend, tau=tau, torder=torder, tvec=tvec, 
            #         svals=svals, res_rk=res_rk, res_rf=res_rf, rkvec=rkvec, etime=etime)
            print(f' dt={tau:.0e}:  etime = {etime:5.2f}   rel error: {rn(X,Xref):.4e}')
        icol += 1
    print('')
    