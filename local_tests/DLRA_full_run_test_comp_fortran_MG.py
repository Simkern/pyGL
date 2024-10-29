#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, copy
import numpy as np
from time import time as tm

from scipy.linalg import expm, qr, svd, svdvals, solve_continuous_lyapunov
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.arnoldi import kryl_expm
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

ffldr = '/home/skern/projects/LightROM/local/TEST'

make_real = True
if_save = True
if_weights = False
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
if (if_weights):
    wc = np.ones(Nxc+2)*dx[0]
else:
    wc = np.ones(Nxc+2)

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
Xref = solve_continuous_lyapunov(L, -Qc)

Qo = C.T @ C @ W
Yref = solve_continuous_lyapunov(L.T, -Qo)

'''
dtv = np.logspace(-6, 0, 13, endpoint=True)
for dt in dtv:
    exptA = expm(dt*L)
    fname = f'exptA_{dt:8.6f}.npy'
    np.save(fname, exptA)

np.save('BBTW.npy', Qc)
sys.exit()
'''
print("\nSVD BBTW:");
ss = svdvals(Qc)[:2]
print(' '.join(f'{x:19.12f}' for x in ss))

print('\nBS solution: X_bs')
print(f' |X_bs|/N   = {np.linalg.norm(Xref)/Nx:16.12f}')
print(f' |res_bs|/N = {CALE(Xref,L,Qc):16.12f}')
print('SVD Xref:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(Xref)[:10]))

nQ    = np.linalg.norm(Qc)
nA    = np.linalg.norm(L)

filenameU = f'CGL_Nx{Nx:02d}_U0_rk_X0_{rk_X0:02d}.npy'
filenameS = f'CGL_Nx{Nx:02d}_S0_rk_X0_{rk_X0:02d}.npy'
fnameU = os.path.join(fldr,filenameU)
fnameS = os.path.join(fldr,filenameS)
print(f"\nRead initial conditions from file:\n  {fnameU}\n  {fnameS}" )
U0 = np.load(fnameU)
S0 = np.load(fnameS)
    
X0    = U0 @ S0 @ U0.T
U0_svd,S0_svd,V0_svdh = svd(X0, full_matrices=False)

print('\nInitial condition: X0')
print(f' |X0|/N  = {np.linalg.norm(X0)/Nx:16.12f}')
print(f' |res|/N = {CALE(X0,L,Qc):16.12f}')
print('SVD:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(X0)[:10]))

# compare RK45 to LR_OSI
tol = 1e-12
Trk   = 0.01
Tend = Trk
tspan = (0,Trk)
Nrep  = 10
tolv  = np.logspace(-12,-12,1)
Tv    = np.linspace(0,Nrep,Nrep+1)*Trk
if (if_weights):
    filename = f'Xrk_CGL_Nx{Nx:02d}_rk0_{rk_X0:02d}_init_W.npz'
else:
    filename = f'Xrk_CGL_Nx{Nx:02d}_rk0_{rk_X0:02d}_init.npz'
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
    print(f"\nRead RK solution from file:\n  {fname}" )
    data = np.load(fname)
    Xrkv = data['Xrkv']
    erel = data['rel_error_Xref']
    Tv   = data['Tv']

nrmX = []
sref = np.zeros((Nrep+1,Nx))
for i in range(Nrep+1):
    nrmX.append(np.linalg.norm(Xrkv[:,:,0,i]))
    _, sref[i,:], _ = np.linalg.svd(Xrkv[:,:,0,i]) 

Tend = 0.01
tol = 1e-12
Xrk = np.squeeze(Xrkv[:,:,tolv==tol,Tv==Tend])

print('\nRKsolution: X_rk')
print(f' |X_rk|/N   = {np.linalg.norm(Xrk)/Nx:16.12f}')
print(f' |res_rk|/N = {CALE(Xrk,L,Qc):16.12f}')
print('SVD:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(Xrk)[:10]))

tau = 0.00001
rk = 20

print('\nLow-rank operator-splitting method for Differential Lyapunov equations.\n')

eps = 1e-12
N = X0.shape[0]
# check rank of inhomogeneity
Sq = svdvals(Qc)
rkq = sum(Sq > eps)
print(f'Numerical rank of inhomogeneity B  ({N:d}x{N:d}): {rkq:3d} (tol={eps:.2e})')

# check rank of initial data
U,S,_ = svd(X0, full_matrices=False)
rk0 = sum(S > eps)
print(f'Numerical rank of initial data  X0 ({N:d}x{N:d}): {rk0:3d} (tol={eps:.2e})')

print('\nMode rank:')
print(f'  rank : {rk:d}')
print(f'  Tend : {Tend:6.4f}')
print(f'  tau  : {tau:6.4f}')
print(f'  TO   : {1:d}')

nchk = 5
    
S0 = np.zeros((rk,rk))
Utmp = np.zeros((N,rk)) #np.random.random_sample((N,rk))
rkmin = min(rk,rk0)
S0[:rkmin,:rkmin] = np.diag(S[:rkmin])
Utmp[:,:rkmin] = U[:,:rkmin]
U0, _ = qr(Utmp, mode='economic')


fnameU = '/home/skern/projects/LightROM/local/CGL_Nx256_U0_rk_20.npy'
fnameS = '/home/skern/projects/LightROM/local/CGL_Nx256_S0_rk_20.npy'
print(f"\nRead extended initial conditions from file:\n  {fnameU}\n  {fnameS}" )
U0 = np.load(fnameU)
S0 = np.load(fnameS)
X0    = U0 @ S0 @ U0.T

'''    
print('\nInitial condition for DLRA: U0 @ S0 @ U0.T')
print(f' |X0|/N  = {np.linalg.norm(X0)/Nx:16.12f}')
print(f' |res|/N = {CALE(X0,L,Qc):16.12f}')
print('SVD:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(X0)[:10]))
'''
print("\nINIT:");
ss = svdvals(S0)[:rk].reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))

'''
fname = os.path.join(ffldr,'GstepKstep_BBTU_init.npy')
print(f"\nRead GstepKstep from file:\n  {fname}" )
BBTUf = np.load(fname)

print('max(BBTU0 (fortran) - Qc @ U0 (python)) = ', (BBTUf - Qc @ U0).max())
 '''    
#
#
# MMMMM
#
#
U0m = U0.copy()
S0m = S0.copy()

nsteps = 4

for j in range(nsteps):
    
    print(f'\n\n\nSTEP {j+1}:\n\n\n')
   
    U1 = expm(tau*L) @ U0m
    
    print('M pre QR: ')
    ip = U1.T @ U1
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8,16)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(16,20)))
    UA, R, p = qr(U1, mode='economic', pivoting = True)
    print(f'\n  Pivoting order M QR:', p,'\n')
    R = R[:,np.argsort(p)]
    #UA, R = qr(U1, mode='economic')
    print('K post QR: Rii:')
    print(' '.join(f'{R[i,i]:16.12f}' for i in range(8)))
    print(' '.join(f'{R[i,i]:16.12f}' for i in range(8,16)))
    print(' '.join(f'{R[i,i]:16.12f}' for i in range(16,20)))
    SA    = R @ S0m @ R.T
    
    '''
    fname = os.path.join(ffldr,'U_after_M.npy')
    print(f"\nRead U after M from file:\n  {fname}" )
    Uam = np.load(fname)
    print('max(U (after M, fortran) - UA (after M python)) = ')
    for i in range(rk):
        print(f'{(Uam[:,i] - UA[:,i]).max():6.3f}', end=' ')
    fname = os.path.join(ffldr,'S_after_M.npy')
    print(f"\nRead A after M from file:\n  {fname}" )
    Sam = np.load(fname)
    print('max(S (after M, fortran) - SA (after M python)) = ', (Sam - SA).max())
    '''
    
    #
    #
    # GGGGG
    #
    #
    
    print("\nSVD  M step:"); 
    rk = UA.shape[1]
    ss = svdvals(SA).reshape((4, int(rk/4)), order='F')
    for i in range(4):
        print(' '.join(f'{x:19.12f}' for x in ss[i,:]))
    print('')
      
    '''
    fname = os.path.join(ffldr,'GstepKstep_BBTU.npy')
    print(f"\nRead GstepKstep from file:\n  {fname}" )
    BBTUf = np.load(fname)
    
    print((BBTUf - Qc @ UA).max())
 
    
    sys.exit() 
    '''
    # solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
    K1 = UA @ SA + tau*(Qc @ UA)
    
    # orthonormalise K1
    print('K pre QR: Rii:')
    ip = K1.T @ K1
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8,16)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(16,20)))
    #U1, Sh = qr(K1, mode='economic')
    U1, Sh, P = qr(K1, pivoting=True, mode='economic')
    print(f'\n  Pivoting order K QR:', P,'\n')
    Sh = Sh[:, np.argsort(P)]
    ip = K1.T @ U1
    print('K1.T @ U1:')
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(8,16)))
    print(' '.join(f'{ip[i,i]:16.12f}' for i in range(16,20)))
    print('K post QR: Rii:')
    print(' '.join(f'{Sh[i,i]:16.12f}' for i in range(8)))
    print(' '.join(f'{Sh[i,i]:16.12f}' for i in range(8,16)))
    print(' '.join(f'{Sh[i,i]:16.12f}' for i in range(16,20)))
    
    if j>0:
        print("\nSVD \t  K step:\t  Sh (1-8)");
        for i in range(20):
            print('\t  ',' '.join(f'{x:15.12f}' for x in Sh[i,:8]))
        print('\t  Sh (9:16)')
        for i in range(20):
            print('\t  ',' '.join(f'{x:15.12f}' for x in Sh[i,8:16]))
        print('\t  Sh (17:20)')
        for i in range(20):
            print('\t  ',' '.join(f'{x:15.12f}' for x in Sh[i,16:]))
        print('\n\t  svd(Sh)')
        ss = svdvals(Sh).reshape((4, int(rk/4)), order='F')
        for i in range(4):
            print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))
    
    # solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
    St = Sh - tau*( U1.T @ Qc @ UA )
    
    if j>0:
        print("\nSVD \t  S step:");
        ss = svdvals(St).reshape((4, int(rk/4)), order='F')
        for i in range(4):
            print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))
    
    # solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
    L1  = St @ UA.T + tau*( U1.T @ Qc )
     
    # update S
    S1  = L1 @ U1
    
    print("\nSVD G step:");
    ss = svdvals(S1).reshape((4, int(rk/4)), order='F')
    for i in range(4):
        print(' '.join(f'{x:19.12f}' for x in ss[i,:]))
    
    U0m = U1.copy()
    S0m = S1.copy()
    
#
#
# DONE
#

print(f"\nEXIT nsteps = {nsteps}:");
ss = svdvals(S1).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))