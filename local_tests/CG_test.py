import numpy as np
import time
import sys

from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator
from matplotlib import pyplot as plt

sys.path.append('..')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.arnoldi import arn, arn_inv
from solvers.lyapunov import lrcfadic_r_gmres

plt.close("all")

def matvec(v):
    Axcount[0] += 1
    return (Lx @ v.reshape(Nxc,2,order='F')).reshape(Nx,1,order='F')

#def applyprec(v):
#    Pxcount[0] += 1
#    return linalg.lu_solve((lu,piv), v)

class cb_res(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk  = []
    def __call__(self, xk=None):
        self.niter += 1
        self.rk.append(linalg.norm(matvec(xk) - b))

class cb_err(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.err   = []
    def __call__(self, x=None):
        self.niter += 1
        self.err.append(linalg.norm(matvec(x)-b))

# Parameters
x0 = -5                      # beginning of spatial domain
x1 = 5                      # end of spatial domain
dx = 0.1                      # Spatial discretisation

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nxc = int(L / dx)                # number of spatial dof
xvec = np.linspace(x0, x1, Nxc)     # spatial grid

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.1
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__   = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,False)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

A = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

Ar = np.real(A)
Ai = np.imag(A)

# make real
A  = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

# Laplacian
Lc = np.asarray(DM2c.todense())

# define linear operators directly via matrices
Lx = LinearOperator(Lc.shape, lambda x: Lc @ x)
# define operators via function calls
L = LinearOperator(matvec=matvec, shape=(Nx,Nx), dtype=Lc.dtype)

b = np.ones((Nx,))

xref = (linalg.solve(Lc,b.reshape(Nxc,2,order='F'))).reshape(Nx,1,order='F')

baseline_counter = cb_res()
etime = time.time()
Axcount = [0]
x, info = sparse.linalg.cg(L, b, tol=1e-12, callback=baseline_counter)
print(time.time() - etime)

fig = plt.figure()

plt.plot(xref, label='xref')
plt.plot(b, label='b')
plt.plot(x, label='x (CG)')
plt.plot(L @ x, label='xref')
plt.legend()

fig2 = plt.figure()

plt.plot(baseline_counter.rk, label='rk')
plt.legend()