import numpy as np
import time

from scipy import linalg as LA
from scipy import optimize as Opt
from scipy import sparse as sp
import sys

from CGL_parameters import *
from CN_integrators import *
from matplotlib import pyplot as plt

plt.close("all")

# Parameters
x0 = -30                      # beginning of spatial domain
x1 = 30                       # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 20                       # Total simulation time
Tmax = 5

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
Nt = int(T / dt)                # number of timesteps
xvec = np.linspace(x0, x1, Nx)     # spatial grid
tvec = np.linspace(0, T, Nt)       # temporal grid

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

L = np.matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)

D,X  = LA.eig(L)

idx = np.argsort(-np.real(D));
D = D[idx]
X = X[:,idx]

# compute Cholesky decomposition of eigenvector inner product
A = X.conj().T @ X
U,S,Vh = LA.svd(A)
s = np.sqrt(S)
F = np.diag(s) @ U.conj().T
Finv = U @ np.diag(np.reciprocal(s))

# retain only relevant modes
imin = -50  # minimum retained imaginary part
i = 0
while np.real(D[i]) > imin:
    i = i + 1
Ds = D[:i]
Xs = X[:,:i]

Nvec = len(Ds)
print(f'Reduced basis uses {Nvec}/{Nx} vectors.\n')
# compute Cholesky decomposition of eigenvector inner product for reduced basis
A = Xs.conj().T @ Xs
U,S,Vh = LA.svd(A)
s = np.sqrt(S)
Fs = np.diag(s) @ U.conj().T
Fsinv = U @ np.diag(np.reciprocal(s))

## testing

T = 10
# Compute matrix exponential
start = time.time()
Phi = LA.expm(T*L)
U,S,Vh = LA.svd(Phi)
Gt = S[0]**2
OIC = Vh[0,:].conj().T
OOC = U[:,0]
etime1 = time.time() - start

start = time.time()
Phi_F = F @ np.diag(np.exp(T*D)) @ Finv
U,S,Vh = LA.svd(Phi_F)
GFt = S[0]**2
etime2 = time.time() - start

start = time.time()
Phi_Fs = Fs @ np.diag(np.exp(T*Ds)) @ Fsinv
U,S,Vh = LA.svd(Phi_Fs)
GFst = S[0]**2
etime3 = time.time() - start

start = time.time()
global_max = Opt.minimize_scalar(lambda t: -LA.norm(Fs @ np.diag(np.exp(t*Ds)) @ Fsinv, ord=2), bounds = [0.1, 50], method='bounded')
etime4 = time.time() - start
Phi = LA.expm(global_max.x*L)
U,S,Vh = LA.svd(Phi)
global_OIC = Vh[0,:].conj().T
global_OOC = U[:,0]
sigma_max = S[0]**2

print(f'Maximum energy amplification:')
print(f'  T = {global_max.x:.2f}')
print(f'  Gt = {sigma_max:.2f}\n')
print(f'Energy amplification at T = {T:.2f}')
print(f'  Gt = {GFst:.2f}\n')

print(f'Comparative cost of the computation of the exponential propagator and svd:')
print(f'  Full operator               ({Nx:>3}x{Nx:>3}): {etime1:.4f} s    ',end='')
print(f's**2 = {Gt:.2f}')
print(f'  Eigenv. expansion           ({Nx:>3}x{Nx:>3}): {etime2:.4f} s    ',end='')
print(f's**2 = {GFt:.2f}')
print(f'  Truncated eigenv. expansion ({Nvec:>3}x{Nvec:>3}): {etime3:.4f} s    ',end='')
print(f's**2 = {GFst:.2f}')

# compute G(t)
nt = 200
tv = np.linspace(0,60,nt)
Gt = [1]
for i,t in enumerate(tv[1:]):
    Gt.append(LA.norm(Fs @ np.diag(np.exp(t*Ds)) @ Fsinv, ord=2)**2)
    
# compute maximum amplification as a function of eigenvector subspace
kvec = np.arange(1,100,2)
Gtk = []
for k in kvec:
    Ds = D[:k]
    Xs = X[:,:k]
    A = Xs.conj().T @ Xs
    U,S,Vh = LA.svd(A)
    s = np.sqrt(S)
    Fs = np.diag(s) @ U.conj().T
    Fsinv = U @ np.diag(np.reciprocal(s))
    Phi_Fs = Fs @ np.diag(np.exp(global_max.x*Ds)) @ Fsinv
    U,S,Vh = LA.svd(Phi_Fs)
    Gtk.append(S[0]**2)
    
fig1 = plt.figure(1)
plt.plot(tv,Gt,color='b')
plt.axvline(x=global_max.x,color='k')
plt.scatter(T,GFst,s=40,c='b')

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(121)
plt.axvline(x=x12,color='k',linestyle='--')
plt.axvline(x=-x12,color='k',linestyle='--')
plt.plot(xvec,np.real(OIC),'r', label='Re')
plt.plot(xvec,np.imag(OIC),'b', label='Im')
plt.plot(xvec,np.abs(OIC),'k', label='|.|')
ax1.title.set_text(f'OptPert for T = {T:.2f}')
ax1 = fig2.add_subplot(122)
plt.axvline(x=x12,color='k',linestyle='--',label='Limits of glob. unstable region')
plt.axvline(x=-x12,color='k',linestyle='--')
plt.plot(xvec,np.real(global_OIC),'r', label='Re')
plt.plot(xvec,np.imag(global_OIC),'b', label='Im')
plt.plot(xvec,np.abs(global_OIC),'k' , label='|.|')
ax1.title.set_text(f'OptPert for T = {global_max.x:.2f} (global)')

fig3 = plt.figure(3)
plt.scatter(kvec,Gtk,s=40,c='b')
plt.xlabel('Number of retained eigenvectors k')
plt.ylabel('Maximum gain G_k(T)')
plt.show()