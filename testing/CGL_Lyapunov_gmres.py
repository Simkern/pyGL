import numpy as np
import sys

from scipy import linalg
from matplotlib import pyplot as plt

sys.path.append('..')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.arnoldi import arn, arn_inv
from solvers.lyapunov import lrcfadic_r_gmres

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

A = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

Ar = np.real(A)
Ai = np.imag(A)

# make real
A = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

D = linalg.eigvals(A)

# compute shifts
b0 = np.ones((Nx,))
# Compute the set of Ritz values R+ for A
na = 60
__,Ha = arn(A,b0,na)
pA,__ = linalg.eig(Ha[0:na,0:na])
pA0   = sorted(pA)
# Compute the set of Ritz values R- for A^-1
nb = 60
__,Hb    = arn_inv(A,b0,nb)
Dbtmp,__ = linalg.eig(Hb[0:nb,0:nb])
pAinv0 = np.array([ 1/r for r in sorted(Dbtmp) ])

pA = np.array([], dtype = complex)
for i,p in enumerate(pA0):
    if np.isreal(p):
        pA = np.append(pA,p)
    else:
        pA = np.append(pA,p)
        pA = np.append(pA,p.conj())
pAinv = np.array([], dtype = complex)
for i,p in enumerate(pAinv0):
    if np.isreal(p):
        pAinv = np.append(pAinv,p)
    else:
        pAinv = np.append(pAinv,p)
        pAinv = np.append(pAinv,p.conj())

idx = np.argsort(-np.real(pA))
pA = pA[idx]
idx = np.argsort(-np.real(pAinv))
pAinv = pAinv[idx]

fig = plt.figure(1)
ax = plt.plot(np.real(pA),np.imag(pA),'ro', mfc='none',  label='Ritz values R+')
plt.plot(np.real(pAinv),np.imag(pAinv),'bo', mfc='none',  label='Ritz values 1/R-')
plt.plot(np.real(D),np.imag(D),'k+', label = 'Eigenvalues A')
plt.axis('square')
plt.legend()
plt.show(block = False)

m = 1

I = np.eye(Nx)

B = np.zeros((Nx,m))
B[-1] = 400

Xref = linalg.solve_continuous_lyapunov(A, -B @ B.T)

npick = 30
pA_v    = np.empty((npick,), dtype='complex')

eps = 1e-12
ip = 0
i = 0
pAnreal = 0
pAncmplx = 0
while ip < npick and i < pA.size:
    p = pA[i]
    if np.abs(np.imag(p)) < eps:
        p = np.real(p)
    if np.real(p) > 0:
        p = -p
    if np.isreal(p):
        if not np.isclose(pA_v, p, atol=eps).any():
            pA_v[ip] = p
            ip    += 1
            pAnreal += 1
    else:
        if not np.isclose(pA_v, p, atol=eps).any() and not np.isclose(pA_v, p.conj(), atol=eps).any():
            pA_v[ip] = p
            ip     += 1
            pAncmplx += 1
    
    i += 1
if i == pA.size:
    print('Not enough shifts to fill pA_v array')
    sys.exit()
print('pA_v:')
print(f'  Number of available shifts:     {pA.size:3d}')
#pvec(pA)
print(f'  Shifts with unique modulus:     {npick:3d},  ',end='')
print(f'  Total:  {pAnreal + 2*pAncmplx:3d},  ',end='')
print(f'  ({pAnreal} real / {pAncmplx} complex conjugate)')
#pvec(pA_v)
    
pAinv_v = np.empty((npick,), dtype='complex')
ip = 0
i = 0
pAinvnreal = 0
pAinvncmplx = 0
while ip < npick and i < pAinv.size:
    p = pAinv[i]
    if np.abs(np.imag(p)) < eps:
        p = np.real(p)
    if np.real(p) > 0:
        p = -p
    if np.isreal(p):
        if not np.isclose(pAinv_v, p, atol=eps).any():
            pAinv_v[ip] = p
            ip    += 1
            pAinvnreal += 1
    else:
        if not np.isclose(pAinv_v, p, atol=eps).any() and not np.isclose(pAinv_v, p.conj(), atol=eps).any():
            pAinv_v[ip] = p
            ip     += 1
            pAinvncmplx += 1
    
    i += 1
if i == pAinv.size:
    print('Not enough shifts to fill PAinv_v array')
    sys.exit()

l = pAinv_v.size
print('pAinv_v:')
print(f'  Number of available shifts:     {pAinv.size:3d}')
#pvec(pAinv)
print(f'  Shifts with unique modulus:     {npick:3d},  ',end='')
print(f'  Total:  {pAnreal + 2*pAncmplx:3d},  ',end='')
print(f'  ({pAnreal} real / {pAncmplx} complex conjugate)')
#pvec(pAinv_v)

#print('Override')
#p_v = np.array([ -0.093, -43.649, -121.179 + 1j*114.858 ])
#p_v = np.array([-0.093, -43.649])
#p_v = np.array([-313.048 -408.700*1j], ndmin = 1)
#p_v = np.array([-303.106 + -390.657*1j, -313.048 -408.700*1j])

tol = 1e-12
gmrestol_v = [ 10**(-i) for i in range(2,13,2) ]
n = 10
nA= 10

fig = plt.figure(2)
pin = np.random.permutation(np.concatenate((pA_v[:nA],pAinv_v[:n-nA])))
for i, gmrestol in enumerate(gmrestol_v):
    print(f'gmres_tol = {gmrestol:.1e}')
    Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = \
        lrcfadic_r_gmres(A, B, pin, tol,'tol', Xref, gmrestol)
    plt.plot(ires,np.log10(nrmx),label=f'tol = {gmrestol:.1e}')
plt.legend()
plt.title(f'{n} CC shifts from K(A,b)')
plt.xlabel('iterations')
plt.ylabel('||X_i - X_ref||_F/ ||X_ref||_F')
plt.show()

fig = plt.figure(3)
nA= 5
pin = np.random.permutation(np.concatenate((pA_v[:nA],pAinv_v[:n-nA])))
for i, gmrestol in enumerate(gmrestol_v):
    print(f'gmres_tol = {gmrestol:.1e}')
    Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = \
        lrcfadic_r_gmres(A, B, pin, tol,'tol', Xref, gmrestol)
    plt.plot(ires,np.log10(nrmx),label=f'tol = {gmrestol:.1e}')
plt.legend()
plt.title(f'{n} CC shifts, {nA} from K(A,b), {n-nA} from K(A^-1,b)')
plt.xlabel('iterations')
plt.ylabel('||X_i - X_ref||_F/ ||X_ref||_F')
plt.show(block=False)