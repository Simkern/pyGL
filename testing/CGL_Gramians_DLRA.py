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

def CALE(X, L, Q, ifadj):
    if ifadj:
        res = L.conj().T @ X + X @ L + Q
    else:
        res = L @ X + X @ L.conj().T + Q
    return res

def BT(X, Y, W):
    # SVD
    Up, Sp, _ = np.linalg.svd(X)
    Uq, Sq, _ = np.linalg.svd(Y)
    # Cholesky
    Pc = Up @ np.diag(np.sqrt(Sp))
    Qc = Uq @ np.diag(np.sqrt(Sq))
    # BT
    Usvd, S, Vsvd = np.linalg.svd(Pc.T @ W @ Qc, full_matrices=False)
    sqrtSinv = np.diag(np.sqrt(1.0/S))
    return sqrtSinv @ Vsvd @ Qc.T, Pc @ Usvd @ sqrtSinv, S

def truncateBT(T,Tinv,S,tol):
    idx = next((i for i,s in enumerate(S) if s<tol))
    return T[:idx,:], Tinv[:,:idx], S[:idx], idx

def freqResponse(A,B,C,W,f):
    I = np.eye(A.shape[0])
    G = np.zeros(len(f), dtype=complex)
    for i, s in enumerate(f):
        G[i] = C[0,:] @ np.linalg.inv(1j*s*I - A) @ W @ B[:,0]
    return G

def make_complex(Xr):
    xr, yr = Xr.shape
    x = int(xr/2)
    y = int(yr/2)
    return Xr[:x,:y] - 1j*Xr[:x,y:]

plt.close('all')

#base = '/home/skern/projects/LightROM/local/'
base = '/home/skern/projects/LightROM/local/Xctl_Yobs_data/data_npy/'

# Parameters
Nx = 128
Nxc = Nx
Nxr = 2*Nx

nf = 40
f = np.linspace(-2,2,nf)

tol = 1e-12

# read
# Operator
Lr = np.load('Lr.npy')
Wr = np.load('W.npy')
W  = Wr[:Nx,:Nx]
Br = np.load('B.npy')
Cr = np.load('C.npy')
L = make_complex(Lr)
B = make_complex(Br)
C = make_complex(Cr)
# Controllability
BBTWr = np.load('BBTW.npy')
BBTW = make_complex(BBTWr)
Xr    = np.load('X_BS.npy')
Xrkvr = np.load('Xrk_W.npy')
X = make_complex(Xr)

sXBS = np.linalg.svd(Xr, full_matrices=False, compute_uv=False)
# Observability
CTCWr = np.load('CTCW.npy')
CTCW = make_complex(CTCWr)
Yr    = np.load('Y_BS.npy')
Yrkvr = np.load('Yrk_W.npy')
Y = make_complex(Yr)

sYBS = np.linalg.svd(Yr, full_matrices=False, compute_uv=False)
# BT
T, Tinv, H = BT(Yr,Xr,Wr)
HBS = np.zeros(Nxr)
TBS, TinvBS, Htrunc, idx = truncateBT(T,Tinv,H,tol)
HBS[:idx] = Htrunc
Abs = TBS @ Wr @ Lr @ TinvBS
Bbs = TBS @ Wr @ Br[:,0:1]
Cbs = Cr[0:1,:] @ Wr @ TinvBS
Gbs = freqResponse(Abs, Bbs, Cbs, W[:idx,:idx], f)

Xrkr = Xrkvr[:,:,-1]
Xrk = make_complex(Xrkr)
sXRK = np.linalg.svd(Xrkr, full_matrices=False, compute_uv=False)
Yrkr = Yrkvr[:,:,-1]
Yrk = make_complex(Yrkr)
sYRK = np.linalg.svd(Yrkr, full_matrices=False, compute_uv=False)
# BT
T, Tinv, H = BT(Xrkr,Yrkr,Wr)
Hrk = np.zeros(Nxr)
Trk, Tinvrk, Htrunc, idx = truncateBT(T,Tinv,H,tol)
Hrk[:idx] = Htrunc
Ark = Trk @ Wr @ Lr @ Tinvrk
Brk = Trk @ Wr @ Br[:,0:1]
Crk = Cr[0:1,:] @ Wr @ Tinvrk
# freqResponse
Grk = freqResponse(Ark, Brk, Crk, W[:idx,:idx], f)



labels = ['Bartels-Stuart','RK']
symbs = ['o','x']

# Read data from DLRA

lbls  = []
dtlbl = []
rklbl = []

TOv = [ 1, 2 ]
Nv  = [ 128 ]
rkv = [ 2, 6, 10, 14, 20, 40 ]
dtv = [ '0.10E+00','0.10E-01','0.10E-02','0.10E-03' ]

DLRAfile = 'DLRA.npy'
basedlra = 'DLRA_'

if not os.path.isfile(DLRAfile):
    XDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr, Nxr), dtype=complex)
    YDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr, Nxr), dtype=complex)
    TDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr, Nxr), dtype=complex)
    TinvDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr, Nxr), dtype=complex)
    SXDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr), dtype=complex)
    SYDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr), dtype=complex)
    HDLRA = np.empty((len(TOv),len(rkv), len(dtv), Nxr), dtype=complex)
    Ad = np.empty((len(TOv),len(rkv), len(dtv), Nxr, Nxr), dtype=complex)
    Bd = np.empty((len(TOv),len(rkv), len(dtv), Nxr, 1), dtype=complex)
    Cd = np.empty((len(TOv),len(rkv), len(dtv), 1, Nxr), dtype=complex)
    Gd = np.empty((len(TOv),len(rkv), len(dtv), nf), dtype=complex)
    for N in Nv:
        for i, torder in enumerate(TOv):
            for j, rk in enumerate(rkv):
                for k, dt in enumerate(dtv):                
                    fnameU = f"data_GLXY_XU_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
                    XU = np.load(base + fnameU)
                    fnameS = f"data_GLXY_XS_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
                    XS = np.load(base + fnameS)
                    fnameU = f"data_GLXY_YU_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
                    YU = np.load(base + fnameU)
                    fnameS = f"data_GLXY_YS_n{N:04d}_TO{torder:d}_rk{rk:02d}_t{dt}.npy"
                    YS = np.load(base + fnameS)
                    Xir = XU @ XS @ XU.T
                    Xi = make_complex(Xir)
                    Yir = YU @ YS @ YU.T
                    Yi = make_complex(Yir)
                    XDLRA[i,j,k,:,:] = Xir
                    YDLRA[i,j,k,:,:] = Yir
                    Sx = np.linalg.svd(Xir, full_matrices=False, compute_uv=False)
                    SXDLRA[i,j,k,:] = Sx
                    Sy = np.linalg.svd(Yir, full_matrices=False, compute_uv=False)
                    SYDLRA[i,j,k,:] = Sy
                    T, Tinv, H = BT(Xir, Yir, Wr)
                    TDLRA[i,j,k,:,:] = T
                    TinvDLRA[i,j,k,:,:] = Tinv
                    HDLRA[i,j,k,:] = H
                    Ad[i,j,k,:,:] = T @ Wr @ Lr @ Tinv
                    Bd[i,j,k,:,0:1] = T @ Wr @ Br[:,0:1]
                    Cd[i,j,k,0:1,:] = Cr[0:1,:] @ Wr @ Tinv
                    idx = min(next((i for i,h in enumerate(H) if h < tol)),rk)
                    Gd[i,j,k,:] = freqResponse(Ad[i,j,k,:idx,:idx], Bd[i,j,k,:idx,0:1], Cd[i,j,k,0:1,:idx], W[:idx,:idx], f)
    np.save(basedlra+'X', XDLRA)
    np.save(basedlra+'Y', YDLRA)
    np.save(basedlra+'sX', SXDLRA)
    np.save(basedlra+'sY', SYDLRA)
    np.save(basedlra+'T', TDLRA)
    np.save(basedlra+'Tinv', TinvDLRA)
    np.save(basedlra+'H', HDLRA)
    np.save(basedlra+'A_BT', Ad)
    np.save(basedlra+'B_BT', Bd)
    np.save(basedlra+'C_BT', Cd)
else:
    XDLRA    = np.load(basedlra+'X')
    YDLRA    = np.load(basedlra+'Y')
    SXDLRA   = np.load(basedlra+'sX')
    SYDLRA   = np.load(basedlra+'sY')
    TDLRA    = np.load(basedlra+'T')
    TinvDLRA = np.load(basedlra+'Tinv')
    HDLRA    = np.load(basedlra+'H')
    Ad       = np.load(basedlra+'A_BT')
    Bd       = np.load(basedlra+'B_BT')
    Cd       = np.load(basedlra+'C_BT')
    
for N in Nv:
    for i, torder in enumerate(TOv):
        for j, rk in enumerate(rkv):
            for k, dt in enumerate(dtv):     
                if i == 0:
                    lbls.append('rk'+str(rk)+' '+dt)
                if i == 0 and j == 0:
                    dtlbl.append(dt)
                if i == 0 and k == 0:
                    rklbl.append(rk)
                    
base = '/home/skern/projects/LightROM/local/'
fname = "GL_Ahat.npy"
Ahat = np.load(base + fname)
fname = "GL_Bhat.npy"
Bhat = np.load(base + fname)
fname = "GL_Chat.npy"
Chat = np.load(base + fname)

#symbs  += ['d','*', 's', '^']
nDLRA = len(XDLRA[0])

print('Check Lyapunov equation residuals :')
print('\n%24s\t\t%12s\t\t%12s\n' %('Controllability:','||res||_2/N','||X||_2/N'))
for (Xi, lbl) in zip([Xr, Xrkr], labels):
    print('%24s\t\t%12.6e\t\t%12.6e' % (lbl, norm(CALE(Xi, Lr, BBTWr, False))/Nxr, norm(Xi)/Nxr))
for i, torder in enumerate(TOv):
    print('\n%24s\t\t%12s\t\t%12s' %('DLRA:  TO rk        dt','||res||_2/N','||X||_2/N'))
    for j, rk in enumerate(rkv):
        for k, dt in enumerate(dtv):
            Xi = XDLRA[i,j,k,:,:]
            print('         %2d %2d  %s\t\t%12.6e\t\t%12.6e' % (torder, rklbl[j], dtlbl[k], norm(CALE(Xi, Lr, BBTWr, False))/Nxr, norm(Xi)/Nxr))
print('\n%24s\t\t%12s\t\t%12s\n' %('Observability:','||res||_2/N','||X||_2/N'))
for (Yi, lbl) in zip([Yr, Yrkr], labels):
    print('%24s\t\t%12.6e\t\t%12.6e' % (lbl, norm(CALE(Yi, Lr, CTCWr, True))/Nxr, norm(Yi)/Nxr))
for i, torder in enumerate(TOv):
    print('\n%24s\t\t%12s\t\t%12s' %('DLRA:  TO rk        dt','||res||_2/N','||X||_2/N'))
    for j, rk in enumerate(rkv):
        for k, dt in enumerate(dtv):
            Yi = YDLRA[i,j,k,:,:]
            print('         %2d %2d  %s\t\t%12.6e\t\t%12.6e' % (torder, rklbl[j], dtlbl[k], norm(CALE(Yi, Lr, CTCWr, True ))/Nxr, norm(Yi)/Nxr))

# Spectra
print('Compute Spectra ... ', end=''); etime = tm()

cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

#
# compare dt
#
iTO = 1
irk = 3   #rkv = [ 2, 6, 10, 14, 20, 40 ]
fig, axs = plt.subplots(2,2)
# spectrum of BS solution
axs[0,0].semilogy(sXBS, color=cc[0])
axs[0,1].semilogy(sYBS, label='BS', color=cc[0])
# spectrum of RK solution
axs[0,0].semilogy(sXRK, color=cc[1])
axs[0,1].semilogy(sYRK, label='RK', color=cc[1])
# difference to BS
axs[1,0].semilogy(abs(sXRK-sXBS), color=cc[1])
axs[1,1].semilogy(abs(sYRK-sYBS), color=cc[1]) 
# DLRA
for i, (dt, lbl) in enumerate(zip(dtv, dtlbl)):
    Sx = SXDLRA[iTO,irk,i,:]
    axs[0,0].semilogy(Sx, color=cc[i+2])
    Sy = SYDLRA[iTO,irk,i,:]
    axs[0,1].semilogy(Sy, label='dt='+lbl, color=cc[i+2])
    # difference to BS
    axs[1,0].semilogy(abs(Sx-sXBS), color=cc[i+2])
    axs[1,1].semilogy(abs(Sy-sYBS), color=cc[i+2])
for ax in axs.flatten():
    ax.set_xlim(0,40)
    ax.set_ylim(tol, 1e3)
axs[0,0].set_title('Controllability')
axs[0,1].set_title('Observability')
axs[1,0].set_title('Absolute difference to BS')
axs[1,1].set_title('Absolute difference to BS')
plt.suptitle('Singular values (TO=%1d, rk=%2d)' % (TOv[iTO], rklbl[irk]))
axs[0,1].legend()

#
# compare rk
#
iTO = 1
idt = 3   #dtv = [ '0.10E+00','0.10E-01','0.10E-02','0.10E-03' ]
fig, axs = plt.subplots(2,2)
# spectrum of BS solution
axs[0,0].semilogy(sXBS, color=cc[0])
axs[0,1].semilogy(sYBS, label='BS', color=cc[0])
# spectrum of RK solution
axs[0,0].semilogy(sXRK, color=cc[1])
axs[0,1].semilogy(sYRK, label='RK', color=cc[1])
# difference to BS
axs[1,0].semilogy(abs(sXRK-sXBS), color=cc[1])
axs[1,1].semilogy(abs(sYRK-sYBS), color=cc[1]) 
# DLRA
for i, (rk, lbl) in enumerate(zip(rkv, rklbl)):
    Sx = SXDLRA[iTO,i,idt,:]
    axs[0,0].semilogy(Sx, color=cc[i+2])
    Sy = SYDLRA[iTO,i,idt,:]
    axs[0,1].semilogy(Sy, label='rk='+str(lbl), color=cc[i+2])
    # difference to BS
    axs[1,0].semilogy(abs(Sx-sXBS), color=cc[i+2])
    axs[1,1].semilogy(abs(Sy-sYBS), color=cc[i+2])
    
for ax in axs.flatten():
    ax.set_xlim(0,40)
    ax.set_ylim(tol, 1e3)
axs[0,0].set_title('Controllability')
axs[0,1].set_title('Observability')
axs[1,0].set_title('Absolute difference to BS')
axs[1,1].set_title('Absolute difference to BS')
plt.suptitle('Singular values (TO=%1d, dt=%s)' % (TOv[iTO], dtlbl[idt]))
axs[0,1].legend()
print(f'done. etime = {tm()-etime}')


# Balanced truncation
print('Compute Balancing Transformations ... ', end=''); etime = tm()



#
# compare dt
#
iTO = 1
irk = 3   #rkv = [ 2, 6, 10, 14, 20, 40 ]
fig, axs = plt.subplots(2,1)
axs[0].semilogy(HBS, label='BS', color=cc[0])
axs[0].semilogy(Hrk, label='RK', color=cc[1])
axs[1].semilogy(abs(Hrk - HBS), color=cc[1])
for i, (dt, lbl) in enumerate(zip(dtv, dtlbl)):
    H = HDLRA[iTO,irk,i,:]
    axs[0].semilogy(H, label=lbl, color=cc[i+2])
    axs[1].semilogy(abs(H-HBS), color=cc[i+2])       
for ax in axs:
    ax.set_xlim(0,40)
    ax.set_ylim(tol, 1e3)
axs[0].set_title('Hankel singular values (TO=%1d, rk=%2d)' % (TOv[iTO], rklbl[irk]))
axs[0].legend()
#
# compare rk
#
iTO = 1
idt = 3   #dtv = [ '0.10E+00','0.10E-01','0.10E-02','0.10E-03' ]
fig, axs = plt.subplots(2,1)
axs[0].semilogy(HBS, label='BS', color=cc[0])
axs[0].semilogy(Hrk, label='RK', color=cc[1])
axs[1].semilogy(abs(Hrk - HBS), color=cc[1])
for i, (rk, lbl) in enumerate(zip(rkv, rklbl)):
    H = HDLRA[iTO,i,idt,:]
    axs[0].semilogy(H, label=lbl, color=cc[i+2])
    axs[1].semilogy(abs(H-HBS), color=cc[i+2])       
for ax in axs:
    ax.set_xlim(0,40)
    ax.set_ylim(tol, 1e3)
axs[0].set_title('Hankel singular values (TO=%1d, dt=%s)' % (TOv[iTO], dtlbl[idt]))
axs[0].legend()
print(f'done. etime = {tm()-etime}')

print('Compute Differential Bode plots ... ', end=''); etime = tm()
#
# compare rk
#
iTO = 1
idt = 3   #dtv = [ '0.10E+00','0.10E-01','0.10E-02','0.10E-03' ]
fig, axs = plt.subplots(2,1)
axs[0].semilogy(f, abs(Gbs), label='BS', color=cc[0])
axs[0].semilogy(f, abs(Grk), label='RK', color=cc[1])
axs[1].semilogy(f, abs(Grk-Gbs), color=cc[1])
for i, (rk, lbl) in enumerate(zip(rkv, rklbl)):
    G = Gd[iTO,i,idt,:]
    axs[0].semilogy(f, abs(G), label=lbl, color=cc[i+2])
    axs[1].semilogy(f, abs(G-Gbs), color=cc[i+2])
plt.title('Transfer functions')
axs[0].legend()
print(f'done. etime = {tm()-etime}')