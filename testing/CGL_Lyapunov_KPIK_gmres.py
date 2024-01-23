import numpy as np
import sys

from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator
from matplotlib import pyplot as plt

sys.path.append('..')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.lyapunov import kpik_gmres

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

plt.close("all")

# Parameters
x0 = -10                      # beginning of spatial domain
x1 = 10                       # end of spatial domain
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

mu_scal,__,__,__,__ = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

A0 = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

# make real
Ar = np.real(A0)
Ai = np.imag(A0)
Amat = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

m = 1
B = np.zeros((Nx,m))
B[-1] = 400



# Laplacian
Lc = np.asarray(gamma*DM2c.todense())
Lr = np.real(Lc)
Li = np.imag(Lc)
L = np.block([[Lr,-Li],[Li,Lr]])

lu,piv = linalg.lu_factor(L)

# define linear operators directly via matrices
Ax = LinearOperator(Amat.shape, lambda x: Amat @ x)
Mx = LinearOperator(Amat.shape, lambda x: linalg.lu_solve((lu,piv), x))
    
# define operators via function calls
A = LinearOperator(matvec=matvec,    shape=Amat.shape, dtype=Amat.dtype)
M = LinearOperator(matvec=applyprec, shape=Amat.shape, dtype=Amat.dtype)
maxiter = 200

normA = np.linalg.norm(A, 'fro')

tol  = 1e-8
tolY = 1e-8
stol = 1e-8
k_max = 100

Z, err2, k_eff, etime, nmatvec = kpik_gmres(A,B,M,k_max,tol,tolY,stol)

print(nmatvec)
