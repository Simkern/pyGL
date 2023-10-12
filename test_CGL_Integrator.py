import numpy as np
import time

from scipy import linalg as LA
from scipy import optimize as Opt
from scipy import sparse as sp
import sys

from git.CGL_parameters import *
from git.CN_integrators import *
from git.diff_mat import *
from git.utils import enorm
from matplotlib import pyplot as plt

plt.close('all')

# Parameters
x0 = -30                      # beginning of spatial domain
x1 = 30                       # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 2                       # Optimisation time
Niter = 10

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

L  = np.matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)
#LH = np.matrix(np.diag(mu) - nu*DM1b + gamma*DM2c).H

# compute optimal via the exponential propagator
Phi = LA.expm(T*L)
U,S,Vh = LA.svd(Phi)
sigma_max = S[0]**2
OOC = U[:,0]
OIC = Vh[0,:].conj().T
print(f'Maximum energy amplification at T = {T:.2f}')
print(f'  Gt = {sigma_max:.2f}\n')

## Test forward integration with OIC

ip0     = enorm(OIC)
q0      = OIC/np.sqrt(ip0)

qt = CN_L_integrate(xvec, tvec, mu, nu, gamma, q0)

print(f'Forward integration of the LCGL system to T = {T:.2f}')
G = enorm(qt[:,-1])
print(f'  Gt = {G:.2f}\n')
    
fig = plt.figure(1)
plt.plot(xvec,np.real(OIC),color='k',linestyle='-',label='q(0)')
plt.plot(xvec,np.real(qt[:,-1])/np.sqrt(G),color='r',linestyle='-',label='q(T)')
plt.plot(xvec,np.real(OOC),color='k',linestyle='--',label='OOC (svd)')
plt.title(f'Direct integration of Optimal perturbation (T = {T:.2f})')
plt.legend()

print(f'Direct-Adjoint loop on the OIC to Topt = {T:.2f}')
## Test adjoint loop using OIC/OOC

# initial condition
E0 = enorm(OIC)
print(f'Initial energy:      E(0) = {E0:.2f}')
q0 = OIC/E0

print(f'Direct ... ',end='')
start = time.time()
q = CN_L_integrate(xvec, tvec, mu, nu, gamma, q0)
etime = time.time() - start
print(f'done:  {etime:.2f}s')
qT = q[:,-1]
ET = enorm(qT)
print(f'Energy at T = {T:.2f}: E(T) = {ET:.2f}')
start = time.time()
psiT = qT/np.sqrt(ET)
print(f'Adjoint ... ',end='')
psi = CN_L_adj_integrate(xvec, tvec, mu, nu, gamma, psiT)
etime = time.time() - start
print(f'done:  {etime:.2f}s')
   
psi0 = psi[:,0] 
E2 = enorm(psi0)
q[:,0] = psi0/np.sqrt(E2)
print(f'Energy at T =  0.00: E(0) = {E2:.2f}\n')
    
Opt = q[:,0]

fig = plt.figure(2)
plt.plot(xvec,np.real(Opt),color='r',label='q(0)_loop')
plt.plot(xvec,np.real(psi[:,-1]),color='b',label='q(T)')
plt.plot(xvec,np.real(OIC),color='k',linestyle='--',label='OIC = q(0)')
plt.plot(xvec,np.real(OOC),color='r',linestyle='--',label='OOC')
plt.legend()
plt.title(f'Direct-Adjoint loop T = {T:.2f}')
plt.show()

print(f'Power iteration to compute optimal growth for Topt = {T:.2f} starting from noise.')
## Test adjoint loop using noise
noise = np.random.randn(Nx,) + 1j*np.random.randn(Nx,)
noise = noise/np.sqrt(enorm(noise))

q0 = noise

for n in range(Niter):
    print(f'Direct ... ', end='')
    start = time.time()
    q = CN_L_integrate(xvec,tvec,mu,nu,gamma,q0)
    qT = q[:,-1]
    G = enorm(qT)
    psiT = qT/np.sqrt(G)
    print(f'Adjoint ... ',end='')
    psi = CN_L_adj_integrate(xvec,tvec,mu,nu,gamma,psiT)
    etime = time.time() - start
    psi0 = psi[:,0]
    q0 = psi0/np.sqrt(enorm(psi0))
    res = abs(G - sigma_max)/G
    print(f'done.   E(T)/E(0) = {G:3.2f}, r = {res:.4e}  {etime:.2f} s')
    
    if res < 1e-12:
        break
    
if res > 1e-12:
    print(f'Maximum number of power iteration steps reached. Relative error = {res:.4e}')
else:
    print(f'Power iteration converged after {n+1} steps.')