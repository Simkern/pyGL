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
from core.utils import p

import solvers.arnoldi as arn
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
Qc = B @ B.T# @ W
Xref = solve_continuous_lyapunov(L, -Qc)

Qo = C.T @ C @ W
Yref = solve_continuous_lyapunov(L.T, -Qo)

nQ    = np.linalg.norm(Qc)
nA    = np.linalg.norm(L)

filenameU = f'CGL_Nx{Nx:02d}_U0_rk_X0_{rk_X0:02d}.npy'
filenameS = f'CGL_Nx{Nx:02d}_S0_rk_X0_{rk_X0:02d}.npy'
fnameU = os.path.join(fldr,filenameU)
fnameS = os.path.join(fldr,filenameS)
if not os.path.isfile(fnameU):
    print("Generate initial conditions")
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = qr(np.random.random_sample((Nx, rk_X0)),mode='economic')
    S0    = np.diag(sorted(s0)[::-1]);
    np.save(fnameU, U0)
    np.save(fnameS, S0)
else:
    print("Read initial conditions from file")
    U0 = np.load(fnameU)
    S0 = np.load(fnameS)
    
X0    = U0 @ S0 @ U0.T
U0_svd,S0_svd,V0_svdh = svd(X0, full_matrices=False)

# compare RK45 to LR_OSI
tol = 1e-12
Trk   = 0.1
Tend = Trk
tspan = (0,Trk)
Nrep  = 10
tolv  = np.logspace(-12,-12,1)
Tv    = np.linspace(0,Nrep,Nrep+1)*Trk
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
    data = np.load(fname)
    Xrkv = data['Xrkv']
    erel = data['rel_error_Xref']
    Tv   = data['Tv']
    
nrmX = []
sref = np.zeros((Nrep+1,Nx))
for i in range(Nrep+1):
    nrmX.append(np.linalg.norm(Xrkv[:,:,0,i]))
    _, sref[i,:], _ = np.linalg.svd(Xrkv[:,:,0,i]) 

Tend = 0.1
tol = 1e-12
Xrk = np.squeeze(Xrkv[:,:,tolv==tol,Tv==Tend])

print('\nRKsolution: X_rk')
print(f' |X_rk|/N   = {np.linalg.norm(Xrk)/Nx:16.12f}')
print(f' |res_rk|/N = {CALE(Xrk,L,Qc):16.12f}')
print('SVD:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(Xrk)[:10]))

rkv = [ 4, 20 ] #[ 4, 8, 20, 40, 60]
tau = [ 0.01 ]
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
print(f'  Chosen rank: {rk:d}\n')

nchk = 5
    
S0 = np.zeros((rk,rk))
Utmp = np.zeros((N,rk)) #np.random.random_sample((N,rk))
rkmin = min(rk,rk0)
S0[:rkmin,:rkmin] = np.diag(S[:rkmin])
Utmp[:,:rkmin] = U[:,:rkmin]
U0, _ = qr(Utmp, mode='economic')

#uTu = U0.T @ U0
#for i in range(rk):
#    print(' '.join(f'{x:6.0e}' for x in uTu[i,:]))
#for i in range(rk):
#    print(f'{i+1}: {}')
    
print('\nInitial condition: X0')
print(f' |X0|/N  = {np.linalg.norm(X0)/Nx:16.12f}')
print(f' |res|/N = {CALE(X0,L,Qc):16.12f}')
print('SVD:')
print('\t\n'.join(f'{x:16.12f}' for x in svdvals(X0)[:10]))

print("\nSVD BBTW:");
ss = svdvals(Qc)[:2]
print(' '.join(f'{x:19.12f}' for x in ss))

#
#  TEST QR M & GK
#
#
#
#

rk  = 20
rk0 = 10
dt  = 0.001
S0 = np.load(os.path.join(ffldr,'M_S0.npy'))

print('\nM Step:\n')

print('\nRegular QR')
exptA_U0 = np.load(os.path.join(ffldr,'M_exptA_U0.npy'))
Q,  R  = qr(exptA_U0, pivoting=False, mode='economic')
Qt, Rt = arn.qr(exptA_U0)
for i in range(Rt.shape[0]):
    if not np.sign(R[i,i]) == np.sign(Rt[i,i]):
        Rt[i,:] *= -1
#p(R, 'R')
#p(Rt,'R LK')
print(f'diff scipy mine: {np.linalg.norm(R-Rt)}')

print('\nPivoted QR')
Qp,  Rp, P  = qr(exptA_U0, pivoting=True, mode='economic')
Qtp, Rtp, Pt = arn.qrp(exptA_U0)
for i in range(Rtp.shape[0]):
    if not np.sign(Rp[i,i]) == np.sign(Rtp[i,i]):
        Rtp[i,:] *= -1
Rp = Rp[:,np.argsort(P)]
Rtp = Rtp[:,np.argsort(Pt)]
#p(Rp, 'R')
#p(Rtp,'R LK')
print(f'diff scipy mine: {np.linalg.norm(Rp-Rtp)}')





print('\nRegular QR M')
exptA_U0 = np.load(os.path.join(ffldr,'M_exptA_U0.npy'))
Q,  R  = qr(exptA_U0, pivoting=False, mode='economic')
Qf = np.load(os.path.join(ffldr,'M_exptA_U0_qr_U.npy'))
Rf = np.load(os.path.join(ffldr,'M_exptA_U0_qr_R.npy'))
flip = np.zeros((R.shape[0]))
for i in range(Rf.shape[0]):
    if not np.sign(R[i,i]) == np.sign(Rf[i,i]):
        R[i,:] *= -1
        flip[i] = 1
#p(R, 'R')
#p(Rf,'R LK')
print('flipped: ', flip)
print(f'diff python fortran: {np.linalg.norm(R-Rf)}')

print('Pivoted QR M')
Qp,  Rp, P  = qr(exptA_U0, pivoting=True, mode='economic')
Qpf = np.load(os.path.join(ffldr,'M_exptA_U0_qrp_U.npy'))
Rpf = np.load(os.path.join(ffldr,'M_exptA_U0_qrp_R.npy'))
Rpfp = np.load(os.path.join(ffldr,'M_exptA_U0_qrp_Rp.npy'))
Pf = np.array([ 1, 4, 3, 2, 5, 6 ]) - 1
flip = np.zeros((R.shape[0]))
for i in range(Rp.shape[0]):
    if not np.sign(Rp[i,i]) == np.sign(Rpf[i,i]):
        Rp[i,:] *= -1
        flip[i] = 1
        #Rpfp[i,:] *= -1
#p(Rp, 'R')
#p(Rpf,'R LK')
print('flipped: ', flip)
print(f'diff python fortran raw:     {np.linalg.norm(Rp-Rpf)}')
Rp = Rp[:,np.argsort(P)]
print(f'diff python fortran pivoted: {np.linalg.norm(Rp-Rpfp)}')
print("Pivots:")
print('\tPython:', P)
print('\tLK    :',Pf)

S  = R @ S0 @ R.T
Sp = Rp @ S0 @ Rp.T
Sf = np.load(os.path.join(ffldr,'M_exptA_U0_qr_S.npy'))
print(f'diff python fortran:         {np.linalg.norm(S-Sf)}')
#p(S, 'S = R S R.T')
#p(Sf,'S LK')
print(f'Sval diff scipy fortran:     {np.linalg.norm(svdvals(S)-svdvals(Sf))}')
print(f'Sval diff piv scipy fortran: {np.linalg.norm(svdvals(Sp)-svdvals(Sf))}')


print('\n\nG Step - K:\n')

K0  = Q @ S
K0f = np.load(os.path.join(ffldr,'GK_K0.npy'))
print('max(K0 - K0f):')
print(' '.join(f'{x:10.8f}' for x in (abs(K0-K0f)).max(axis=0)))

K1  = K0 + dt*(Qc @ Q)
K1_  = K0f + dt*(Qc @ Qf)
K1f = np.load(os.path.join(ffldr,'GK_K1.npy'))
print('max(K1 - K1f):')
print(' '.join(f'{x:10.8f}' for x in (abs(K1_-K1f)).max(axis=0)))
print(' max val:')
print(' '.join(f'{x:10.8f}' for x in (abs(K1_)).max(axis=0)))
print('Max perturbation of K0:', (dt*Qc @ Qp).max())

KQ, KR = qr(K1_, pivoting=False, mode='economic')
KQf   = np.load(os.path.join(ffldr,'GK_qr_K1_U.npy'))
KRf   = np.load(os.path.join(ffldr,'GK_qr_K1_R.npy'))
flip = np.zeros((R.shape[0]))
for i in range(KR.shape[0]):
    if not np.sign(KR[i,i]) == np.sign(KRf[i,i]):
        KR[i,:] *= -1
        flip[i] = 1
print('flipped: ', flip)
print(f'diff python fortran:     {np.linalg.norm(KR-KRf)}')
sys.exit()








print('\nG Step - K:\n')

K0 = Qp @ Sp
K0f    = np.load(os.path.join(ffldr,'GK_K0.npy')) # = Qpf @ Sf
print('max(K0 - K0f):')
print(' '.join(f'{x:10.8f}' for x in (K0-K0f).max(axis=0)))

K1 = K0 + dt*(Qc @ Qp)
K1_ = K0f + dt*(Qc @ Qpf)
K1f    = np.load(os.path.join(ffldr,'GK_K1.npy'))
print('max(K1 - K1f):')
print(' '.join(f'{x:10.8f}' for x in (K1-K1f).max(axis=0)))#
print('Max perturbation of K0:', (Qc @ Qp).max(), f' x dt ({dt:6.4f})')

print('Regular QR GK')
KQ, KR = qr(K1, pivoting=False, mode='economic')
KQf   = np.load(os.path.join(ffldr,'GK_qr_K1_U.npy'))
KRf   = np.load(os.path.join(ffldr,'GK_qr_K1_R.npy'))
p(KR, 'KR')
p(KRf,'KR LK')
p(KR-KRf,'diff')
for i in range(rk0):
    print(f' row {i+1}: max err: {(KR-KRf)[i,:].max()}')
for i in range(rk0,rk-1):
    print(f' row {i+1}: max err: {(KR-KRf)[i,i+1:].max()}')

print('Pivoted QR GK')
KQp, KRp, KP = qr(K1, pivoting=True, mode='economic')
KRp = KRp[:,np.argsort(KP)]
KQfp  = np.load(os.path.join(ffldr,'GK_qrp_K1_U.npy'))
KRfp  = np.load(os.path.join(ffldr,'GK_qrp_K1_R.npy'))
KRpfp = np.load(os.path.join(ffldr,'GK_qrp_K1_Rp.npy'))
KPf = np.array([ 1, 4, 3, 2, 5, 6 ]) - 1
p(KRp, 'R')
p(KRpfp,'R LK')
p(KRp-KRpfp,'diff')
for i in range(rk0):
    print(f' row {i+1}: max err: {(KRp-KRpfp)[i,:].max()}')
for i in range(rk0,rk-1):
    print(f' row {i+1}: max err: {(KRp-KRpfp)[i,i+1:].max()}')
print("Pivots:")
print('Python:', KP)
print('LK    :',KPf)


sys.exit()

U0f = np.load(os.path.join(ffldr,'U0.npy'))
U0fw = np.load(os.path.join(ffldr,'U0W.npy'))
print("\n%18s %18s %18s %18s %18s" % ("Python U0", "Fortran U0", "error", "Fortran U0 W", "error"))
for i, (U, u2, u3, u4, u5) in enumerate(zip(U0[:nchk,0], U0f[:nchk,0], U0f[:nchk,0]-U0[:nchk,0], U0fw[:nchk,0], U0fw[:nchk,0]-U0[:nchk,0])):
    print(' '.join(f'{x:18.12f}' for x in [ U, u2, u3, u4, u5 ]))
print("\nPython U0 [1, 2, 3, 4]")
for i in range(5):
    print(' '.join(f'{x:19.12f}' for x in U0[i,:rk]))
    
# applying outer product
print('\nOuter product')
print("\nSVD python:");
Utest, Stest = qr(Qc @ U0, pivoting=False, mode='economic')
ss = svdvals(Stest).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))
print("\nSVD python:");
BBTWU0f = np.load(os.path.join(ffldr,'BBTW_U0.npy'))
Utest, Stest = qr(BBTWU0f, pivoting=False, mode='economic')
ss = svdvals(Stest).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))

# using fortran initial data now!
    
print("\n%-20s %-20s %-20s %-20s" % ("Python Matrix", "Python kexpm", "Fortran RK", "Fortran kexpm"))
Um = expm(tau*L) @ U0f
Uk = kryl_expm(L,U0f,10,tau)
Ufr = np.load(os.path.join(ffldr,'exptA_RK.npy'))
Ufk = np.load(os.path.join(ffldr,'exptA_kexpm.npy'))
for U, u2, u3, u4 in zip(Um[:nchk,0], Uk[:nchk,0], Ufr[:nchk,0], Ufk[:nchk,0]):
    print(' '.join(f'{x:19.12f}' for x in [ U, u2, u3, u4 ]))
print("First element:")
for U, u2, u3, u4 in zip(Um[0,:], Uk[0,:], Ufr[0,:], Ufk[0,:]):
    print(' '.join(f'{x:19.12f}' for x in [ U, u2, u3, u4 ]))
    
print("Errors wrt python matrix:")
for U, u2, u3, u4 in zip(Um[:nchk,0], Uk[:nchk,0], Ufr[:nchk,0], Ufk[:nchk,0]):
    print(' '.join(f'{x:19.12f}' for x in [ U-U, u2-U, u3-U, u4-U ]))
print("Max error wrt python matrix:")
for i in range(rk):
    max_values = (Um[:,i] - Um[:,i]).max(), (Uk[:,i] - Um[:,i]).max(), (Ufr[:,i] - Um[:,i]).max(), (Ufk[:,i] - Um[:,i]).max()
    # Unpack the maximum values
    U, u2, u3, u4 = max_values
    print(' '.join(f'{x:19.12f}' for x in [ U, u2, u3, u4 ]),f'vector{i+1:2d}')

print("\nQR decomposition:")
Q,  R  = qr(Um, pivoting=False, mode='economic')
Qp, Rp, P = qr(Um, pivoting=True, mode='economic'); Rp = Rp[:,np.argsort(P)]
# python QR on fortran data --> same as python data
#Qf, Rf = qr(Ufr, pivoting=False, mode='economic')
#Qf, Rf, Pf = qr(Ufr, pivoting=True, mode='economic'); Rf = Rf[:,np.argsort(Pf)]
#Qfp, Rfp = qr(Ufk, pivoting=False, mode='economic')
#print("\n:python QR on fortran data")
#for i in range(rk):
#    print(' '.join(f'{x:8.5f}' for x in Rf[i,:]))
# LightKrylov QR on fortran data --> error
Rf  = np.load(os.path.join(ffldr,'qr_R.npy'))
Rfp = np.load(os.path.join(ffldr,'qrp_R.npy'))
#print("\n:LightKrylov QR on fortran data")
#for i in range(rk):
#    print(' '.join(f'{x:8.5f}' for x in Rf[i,:]))

for (s, sp, sf, sfp) in zip(svdvals(R), svdvals(Rp), svdvals(Rf), svdvals(Rfp)):
        print(' '.join(f'{x:19.12f}' for x in [ s, sp, sf, sfp ]))
S  = R @ S0 @ R.T
Sp = Rp @ S0 @ Rp.T
Sf = np.load(os.path.join(ffldr,'S_RSRT.npy'))
print("\nSVD (R @ S @ R.T):")
for (s, sp, sf) in zip(svdvals(S), svdvals(Sp), svdvals(Sf)):
        print(' '.join(f'{x:19.12f}' for x in [ s, sp, sf ]))

UA = Q

print("\nSVD G step before K:"); 
ss = svdvals(S).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))
# solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
K1 = UA @ S + tau*(Qc @ UA)

# orthonormalise K1
U1, Sh = qr(K1, mode='economic')

print("\nSVD \t  K step:");
ss = svdvals(Sh).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))

# solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
St = Sh - tau*( U1.T @ Qc @ UA )

print("\nSVD \t  S step:");
ss = svdvals(St).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))
 
# solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
L1  = St @ UA.T + tau*( U1.T @ Qc )
 
# update S
S1  = L1 @ U1

print("\nSVD L step:");
ss = svdvals(S1).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))
    
U1 = expm(tau*L) @ U1

UA, R, p = qr(U1, mode='economic', pivoting = True)
R = R[:,np.argsort(p)]
SA    = R @ S1 @ R.T

print("\nSVD  M step:"); 
rk = UA.shape[1]
ss = svdvals(SA).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))
    
# solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
K1 = UA @ SA + tau*(Qc @ UA)

# orthonormalise K1
U1, Sh = qr(K1, mode='economic')

print("\nSVD \t  K step:");
ss = svdvals(Sh).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))

# solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
St = Sh - tau*( U1.T @ Qc @ UA )

print("\nSVD \t  S step:");
ss = svdvals(St).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print('\t  ',' '.join(f'{x:19.12f}' for x in ss[i,:]))
 
# solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
L1  = St @ UA.T + tau*( U1.T @ Qc )
 
# update S
S1  = L1 @ U1

print("\nSVD L step:");
ss = svdvals(S1).reshape((4, int(rk/4)), order='F')
for i in range(4):
    print(' '.join(f'{x:19.12f}' for x in ss[i,:]))