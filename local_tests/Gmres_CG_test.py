import numpy as np
import time
import sys

from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator
from matplotlib import pyplot as plt

from CGLmat import getCGLmatrices 
from lyap_shifts import compute_shifts

sys.path.append('..')

from git.solvers.arnoldi import arn, arn_inv
from git.core.diff_mat import FDmat
from git.core.CGL_parameters import CGL, CGL2

#from git.solvers.lyapunov import lrcfadic_r_gmres

plt.close("all")


class cb_res(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk = []

    def __call__(self, rk=None):
        self.niter += 1
        self.rk.append(rk)


class cb_err(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.err = []

    def __call__(self, x=None):
        self.niter += 1
        self.err.append(linalg.norm(matvec(x)-b))

A,L,Lc,Nx,Nxc = getCGLmatrices()
pA, pAinv, pA_v, pAinv_v = compute_shifts(60, 60, A, Nx, 8)

def matvec(v):
    Axcount[0] += 1
    return A @ v


def Ashift(v, ip):
    Axcount[0] += 1
    return Avec[ip] @ v


def matvecshiftprec(v, p):
    mvPxcount[-1] += 1
    vr = v.reshape(Nxc, 2, order='F')
    x = Lc @ vr - p * vr
    return x.reshape(Nx, 1, order='F')


def matvecprec(v):
    mvPxcount[-1] += 1
    return (Lc @ v.reshape(Nxc, 2, order='F')).reshape(Nx, 1, order='F')


def applyprec(v):
    Pxcount[0] += 1
    mvPxcount.append(0)
    x, info = sparse.linalg.cg(opL, v, tol=1e-9, callback=prec_counter)
    return x


#def applyshiftprec(v):
#    Pxcount[0] += 1
#    mvPxcount.append(0)
#    x, info = sparse.linalg.cg(opshiftL, v, tol=1e-9, callback=prec_counter)
#    return x


# generate shifted real matrices
Avec = []
I = np.eye(A.shape[0])
for ip, p in enumerate(pA_v):
    if np.isreal(p):
        Atmp = A + np.real(p)*I
    else:
        s_i = 2*np.real(p)
        t_i = np.abs(p)**2
        Atmp = A @ A + s_i*A + t_i*I
    Avec.append(Atmp)

b = np.ones((Nx,))

# define operators via function calls
opA    = LinearOperator(matvec=matvec,     shape=A.shape, dtype=A.dtype)
opL    = LinearOperator(matvec=matvecprec, shape=A.shape, dtype=A.dtype)
opLinv = LinearOperator(matvec=applyprec,  shape=A.shape, dtype=A.dtype)

# define linear operators directly via matrices
#Ax = LinearOperator(A.shape, lambda x:  A @ x)
#Lx = LinearOperator(Lc.shape, lambda x: Lc @ x)




maxiter = 100

baseline_counter = cb_res()
prec_counter = cb_res()
etime = time.time()
Axcount = [0]
Pxcount = [0]
mvPxcount = []
x, info = sparse.linalg.gmres(opA, b, tol=1e-9,
                              callback=baseline_counter, callback_type='pr_norm',
                              restart=20, maxiter=maxiter)
res = linalg.norm(matvec(x) - b)
print(f'No frills GMRES:  \t      info : {info}', end='')
if info == 0:
    print(f'  Converged in {baseline_counter.niter} iterations.')
else:
    print('')
print(f'  res   = {res:.4e}')
print(f'  etime = {time.time() - etime:8.6f}')
print(f'  Ax    = {Axcount[0]}')


lap40_counter_res = cb_res()
etime = time.time()
Axcount = [0]
Pxcount = [0]
mvPxcount = [0]
x, info = sparse.linalg.gmres(opA, b, M=opLinv, tol=1e-9,
                              callback=lap40_counter_res, callback_type='pr_norm',
                              restart=40, maxiter=maxiter)
print(f'Laplacian P-GMRES(40):\t  info : {info}', end='')
if info == 0:
    print(f' -->  Converged in {lap40_counter_res.niter} iterations.')
else:
    print('')
print(f'  res   = {linalg.norm(matvec(x) - b):.4e}')
print(f'  etime = {time.time() - etime:8.6f}')
print(f'  Ax    = {Axcount[0]}')
print(f'  Px    = {Pxcount[0]}')

lap20_counter_res = cb_res()
etime = time.time()
Axcount = [0]
Pxcount = [0]
x, info = sparse.linalg.gmres(opA, b, M=opLinv, tol=1e-9,
                              callback=lap20_counter_res, callback_type='pr_norm',
                              restart=20, maxiter=maxiter)
print(f'Laplacian P-GMRES(20):\t  info : {info}', end='')
if info == 0:
    print(f'  Converged in {lap20_counter_res.niter} iterations.')
else:
    print('')
print(f'  res   = {linalg.norm(matvec(x) - b):.4e}')
print(f'  etime = {time.time() - etime:8.6f}')
print(f'  Ax    = {Axcount[0]}')
print(f'  Px    = {Pxcount[0]}')

lap10_counter_res = cb_res()
etime = time.time()
Axcount = [0]
Pxcount = [0]
x, info = sparse.linalg.gmres(opA, b, M=opLinv, tol=1e-9,
                              callback=lap10_counter_res, callback_type='pr_norm',
                              restart=10, maxiter=maxiter)
print(f'Laplacian P-GMRES(10):\t  info : {info}', end='')
if info == 0:
    print(f'  Converged in {lap10_counter_res.niter} iterations.')
else:
    print('')
print(f'  res   = {linalg.norm(matvec(x) - b):.4e}')
print(f'  etime = {time.time() - etime:8.6f}')
print(f'  Ax    = {Axcount[0]}')
print(f'  Px    = {Pxcount[0]}')

fig = plt.figure(1)
plt.plot(baseline_counter.rk, label='baseline GMRES (no preconditioning)')
#plt.plot(np.arange(6)*20, lap_counter_err.err)
plt.plot(lap10_counter_res.rk, label='L-preconditioned GMRES(10)')
plt.plot(lap20_counter_res.rk, label='L-preconditioned GMRES(20)')
plt.plot(lap40_counter_res.rk, label='L-preconditioned GMRES(40)')
plt.xlim([0, 500])
plt.yscale('log')
plt.legend()

sys.exit()
# direct shifted problem
fig = plt.figure(2)
plt.subplot(121)
for ip, p in enumerate(pA_v):
    gmres40shift_res = cb_res()
    etime = time.time()
    Axcount = [0]
    Pxcount = [0]
    mvPxcount = [0]
    opAs = LinearOperator(matvec=lambda v: Ashift(v, ip),
                          shape=A.shape, dtype=A.dtype)
    x, info = sparse.linalg.gmres(opAs, b, tol=1e-6,
                                  callback=gmres40shift_res,
                                  callback_type='pr_norm',
                                  restart=40, maxiter=maxiter)
    print(f'GMRES(40): p = {pA_v[ip]:f}\n\t  info : {info}', end='')
    if info == 0:
        print(f' -->  Converged in {gmres40shift_res.niter} iterations.')
    else:
        print('')
    plt.plot(gmres40shift_res.rk, label=f'ip = {pA_v[ip]:6.4f}')
plt.yscale('log')
plt.xlim((0, 500))
plt.title('GMRES(40)')
# plt.show(block=False)

# preconditioning!
'''
plt.subplot(132)
for ip, p in enumerate(pA_v):
    lap40shift_res = cb_res()
    etime = time.time()
    Axcount = [0]
    Pxcount = [0]
    mvPxcount = [0]
    pabs = np.abs(p)**2
    opshiftL = LinearOperator(matvec=lambda v: matvecshiftprec(
        v, pabs), shape=A.shape, dtype=A.dtype)
    opshiftLinv = LinearOperator(
        matvec=applyshiftprec, shape=A.shape, dtype=Lc.dtype)
    opAs = LinearOperator(matvec=lambda v: Ashift(v, ip),
                          shape=A.shape, dtype=A.dtype)
    x, info = sparse.linalg.gmres(opAs, b, M=opshiftLinv, tol=1e-9,
                                  callback=lap40shift_res,
                                  callback_type='pr_norm',
                                  restart=40, maxiter=maxiter)
    print(
        f'Shifted Laplacian P-GMRES(40): p = {pA_v[ip]:f}\n\t  info : {info}', end='')
    if info == 0:
        print(f' -->  Converged in {lap40shift_res.niter} iterations.')
    else:
        print('')
    plt.plot(lap40shift_res.rk, label=f'ip = {pA_v[ip]:4.1f}')
plt.yscale('log')
plt.xlim((0, 500))
plt.title('Sh L-prec GMRES(40)')
'''

#fig = plt.figure(3)
plt.subplot(122)
for ip, p in enumerate(pA_v):
    lap40shift_res = cb_res()
    etime = time.time()
    Axcount = [0]
    Pxcount = [0]
    mvPxcount = [0]
    opAs = LinearOperator(matvec=lambda v: Ashift(v, ip),
                          shape=A.shape, dtype=A.dtype)
    x, info = sparse.linalg.gmres(opAs, b, M=opLinv, tol=1e-6,
                                  callback=lap40shift_res,
                                  callback_type='pr_norm',
                                  restart=40, maxiter=maxiter)
    print(
        f'Laplacian P-GMRES(40): p = {pA_v[ip]:f}\n\t  info : {info}', end='')
    if info == 0:
        print(f' -->  Converged in {lap40shift_res.niter} iterations.')
    else:
        print('')
    plt.plot(lap40shift_res.rk, label=f'ip = {pA_v[ip]:4.1f}')
plt.yscale('log')
plt.legend()
plt.xlim((0, 500))
plt.title('L-prec GMRES(40)')
plt.show(block=False)