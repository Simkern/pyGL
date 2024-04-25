import numpy as np
import time
import sys

from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, gmres, cg
from matplotlib import pyplot as plt

sys.path.append('..')

from git.core.CGL_parameters import CGL, CGL2
from git.core.diff_mat import FDmat

from git.solvers.arnoldi import arn, arn_inv
from git.solvers.lyapunov import lrcfadic_r_gmres

from CGLmat import getCGLmatrices
from wrappers import gmres_wrapper, pgmres_lu, cg_wrapper, pgmres_cg

plt.close("all")

def matvecL(v):
    return (Lc @ x.reshape(Nxc, 2, order='F')).reshape(Nx, 1, order='F')

def matvec(v):
    Axcount[0] += 1
    return A @ v

def applyprec(v):
    Pxcount[0] += 1
    return linalg.lu_solve((lu,piv), v)

class gmres_res(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk  = []
    def __call__(self, rk=None):
        self.niter += 1
        self.rk.append(rk)

class gmres_err(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.err   = []
    def __call__(self, x=None):
        self.niter += 1
        self.err.append(linalg.norm(matvec(x)-b))

A,L,Lc,Nx,Nxc = getCGLmatrices()

R = L - np.diagonal(600*np.ones((Nx,1)))

E,V = np.linalg.eig(A)

P = np.linalg.inv(-R) @ A
E1,V1 = np.linalg.eig(P)
E2,V2 = np.linalg.eig(-R)

plt.figure(1)
plt.scatter(np.real(E),np.imag(E),label='A')
plt.scatter(np.real(E1),np.imag(E1),label='-L^{-1} A')
plt.scatter(np.real(E2),np.imag(E2),label='-L')
plt.ylim(-5,5)
plt.xlim(-5, 2)
plt.legend()

lu,piv = linalg.lu_factor(-L)

b = np.random.rand(Nx,)
#b = np.ones((Nx,))
   
# define operators via function calls
opA = LinearOperator(matvec=matvec, shape=A.shape, dtype=A.dtype)
opM = LinearOperator(matvec=applyprec, shape=A.shape, dtype=A.dtype)
maxiter = 150

x, info, Axcount, res, niter = gmres_wrapper(A, b,
                                    tol=1e-6, 
                                    restart = 200,
                                    maxiter=maxiter)

x, info, Axcount, Pxcount, res2, niter = pgmres_lu(A, b, -L, 
                                    tol=1e-6, 
                                    restart = 20,
                                    maxiter=maxiter)
sols = []
res3 = []
#plt.figure(2)
def cg_cb(xk):
    sols.append(xk)
    res3.append(np.linalg.norm(-L @ xk - b))
    #if len(sols) % 5 == 0:
    #    plt.plot(xk)

x, info = cg(-L,b,tol=1e-8,maxiter=maxiter,callback=cg_cb)

x, info, Axount, Pxcount, Mxcount, res4, pres4 = pgmres_cg(
                                    A, b, -L,
                                    tol=1e-6,
                                    ptol=1e-8,
                                    restart = 20,
                                    maxiter=maxiter)

plt.figure(3)
plt.semilogy(res,label='GMRES')
plt.semilogy(res2,label='PGMRES LU')
plt.semilogy(res3,label='CG')
plt.semilogy(res4,label='PGMRES CG')
plt.legend()
'''
plt.figure(2)
for ires, r in enumerate(Pres):
    plt.semilogy(r)
    print(f'Iteration {ires+1}: final residual {r[-1]}')
'''
sys.exit()

x, info = sparse.linalg.gmres(opA,b, tol=1e-9, \
                              callback=baseline_counter, callback_type='pr_norm', \
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

'''
lap_counter_err = gmres_err()
etime = time.time()
Axcount = [0]     
Pxcount = [0]  
x, info = sparse.linalg.gmres(Ax,b, M=Mx, tol=1e-9, \
                              callback=lap_counter_err, callback_type='x', \
                              restart=20, maxiter=maxiter)
'''

lap10_counter_res = gmres_res()
etime = time.time()
Axcount = [0]     
Pxcount = [0]  
x, info = sparse.linalg.gmres(opA,b, M=opM, tol=1e-9, \
                              callback=lap10_counter_res, callback_type='pr_norm', \
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

lap20_counter_res = gmres_res()
etime = time.time()
Axcount = [0]     
Pxcount = [0]  
x, info = sparse.linalg.gmres(opA,b, M=opM, tol=1e-9, \
                              callback=lap20_counter_res, callback_type='pr_norm', \
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

lap40_counter_res = gmres_res()
etime = time.time()
Axcount = [0]     
Pxcount = [0]  
x, info = sparse.linalg.gmres(opA,b, M=opM, tol=1e-9, \
                              callback=lap40_counter_res, callback_type='pr_norm', \
                              restart=40, maxiter=maxiter)
print(f'Laplacian P-GMRES(40):\t  info : {info}', end='')
if info == 0:
    print(f'  Converged in {lap40_counter_res.niter} iterations.')
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



'''
print('A - lambda*I from K(A,b)')
etime = time.time()
x, exitcode = sparse.linalg.gmres(A - pA[0]*I,b,atol=1e-6, restart=20, maxiter=20)
print(linalg.norm(A @ x - b))
print(f'{exitcode}: {time.time() - etime}')
print('A - lambda*I from K(A^-1,b)')
etime = time.time()
x, exitcode = sparse.linalg.gmres(A - pAinv[0]*I,b,atol=1e-6, restart=20, maxiter=20)
print(linalg.norm(A @ x - b))
print(f'{exitcode}: {time.time() - etime}')
'''

