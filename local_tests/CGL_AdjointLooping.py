import numpy as np
import time
import sys

from scipy import linalg as LA
from matplotlib import pyplot as plt

sys.path.append('git')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
from solvers.CN_integrators import CN_L_integrate, CN_L_adj_integrate, CN_L_advance, CN_L_adj_advance


plt.close('all')

# Parameters
x0 = -30                      # beginning of spatial domain
x1 = 30                       # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 4                       # Optimisation time
Niter = 5

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

# =============================================================================
# #Test forward integration with OIC
# 
# ip0     = np.dot(OIC.conj(),OIC)
# q0      = OIC/np.sqrt(ip0)
# 
# qt = CN_L_integrate(xvec, tvec, mu, nu, gamma, q0)
# 
# G = np.real(np.inner(qt[:,-1].conj(),qt[:,-1]))
# print(f'  Gt = {G:.2f}\n')
#     
# fig = plt.figure()
# plt.plot(xvec,np.real(OIC),color='k',linestyle='-')
# plt.plot(xvec,np.real(qt[:,-1])/np.sqrt(G),color='r',linestyle='-')
# plt.plot(xvec,np.real(OOC),color='k',linestyle='--')
# 
# sys.exit()
# =============================================================================

# Test adjoint loop using OIC/OOC

# initial condition
E0 = np.real(np.dot(OIC.conj(),OIC))
print(f'Initial energy:      E(0) = {E0:.2f}')
q0 = OIC/E0

print('Direct ... ',end='')
start = time.time()
q = CN_L_integrate(xvec, tvec, mu, nu, gamma, q0)
etime = time.time() - start
print(f'done:  {etime:.2f}s')
qT = q[:,-1]
ET = np.real(np.dot(qT.conj().T,qT))
print(f'Energy at T = {T:.2f}: E(T) = {ET:.2f}')
start = time.time()
psiT = qT/np.sqrt(ET)
print('Adjoint ... ',end='')
psi = CN_L_adj_integrate(xvec, tvec, mu, nu, gamma, psiT, q)
etime = time.time() - start
print(f'done:  {etime:.2f}s')
   
psi0 = psi[:,0] 
E2 = np.real(np.dot(psi0.conj().T,psi0))
q[:,0] = psi0/np.sqrt(E2)
print(f'Energy at T =  0.00: E(0) = {E2:.2f}')
    
Optimal = q[:,0]

fig = plt.figure(1)
plt.plot(xvec,np.real(Optimal),color='r',label='q(0)_loop')
plt.plot(xvec,np.real(psi[:,-1]),color='b',label='q(T)')
plt.plot(xvec,np.real(OIC),color='k',linestyle='--',label='OIC = q(0)')
plt.plot(xvec,np.real(OOC),color='k',linestyle='--',label='OOC')
plt.legend()
plt.title(f'Direct-Adjoint loop T = {T:.2f}')
plt.show()

sys.exit()
    
q0 = np.random.randn(Nx,) + 1j*np.random.randn(Nx,)
q0 = q0/np.sqrt(np.dot(q0.conj().T,q0)) # normalize

q   = np.zeros((Nx,Nt), dtype=complex)
psi = np.zeros((Nx,Nt), dtype=complex)

# initial condition
q[:,0] = q0

cs = np.linspace(1,0,Niter+2)
cs = cs[1:-1]

fig = plt.figure(1)
plt.plot(xvec,np.real(q0),color='k',label='IC',alpha=0.1)

for n in range(Niter):
    
    print(f'Run {n:>3}: Direct ... ',end='')
    start = time.time()
    for it in range(Nt-1):
        q[:,it+1] = CN_L_advance(q[:,it],L,dt) 
        
    qT = q[:,-1]
    psi[:,-1] = qT/np.sqrt(np.dot(qT.conj().T,qT))
    
    print('Adjoint ... ',end='')
    for it in reversed(range(Nt-1)):
        psi[:,it] = CN_L_adj_advance(psi[:,it+1],L.H,q[:,it],q[:,it+1],dt)
    etime = time.time() - start
    print(f'done:  {etime:.2f}s')
    
    psi0 = psi[:,0]
    q[:,0] = psi0/np.sqrt(np.dot(psi0.conj().T,psi0))

    if not n == Niter-1:
        plt.plot(xvec,np.real(q[:,0]),color=str(cs[n]),label='opt '+str(n+1))

Optimal = q[:,0]
plt.plot(xvec,np.real(Optimal),color='r',label='Opt')
plt.plot(xvec,np.real(OIC),color='b',linestyle='--',label='OIC')
#plt.plot(xvec,np.imag(OIC),color='r',linestyle='--',label='i')
#plt.plot(xvec,np.abs(OIC),color='k',linestyle='--',label='OIC')
plt.legend()
plt.show()
