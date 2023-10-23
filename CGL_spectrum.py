import numpy as np
from scipy import linalg as LA

from CGL_parameters import *
from diff_mat import *
from matplotlib import pyplot as plt

# Parameters
x0 = -30                       # beginning of spatial domain
x1 = 30                         # end of spatial domain
dx = 0.1                      # Spatial discretisation
dt = 0.01                       # Time step
T  = 20                         # Total simulation time

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
Nt = int(T / dt)                # number of timesteps
x = np.linspace(x0, x1, Nx)     # spatial grid
t = np.linspace(0, T, Nt)       # temporal grid

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.1
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu,nu,gamma,Umax,mu_t = CGL2(x,mu0,cu,cd,U,mu2,True)

h = np.sqrt(-2*mu2*gamma)
mu_c = mu_t + np.abs(h)/2*np.cos(np.angle(gamma)/2)

neigs = 100
l = np.zeros((neigs,),dtype='complex')

for n in range(neigs):
    l[n] = (mu0 - cu**2) - (nu**2/(4*gamma)) - (n + 0.5)*h

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(x)

L  = np.matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)
Lb = np.matrix(np.diag(mu) - nu*DM1b + gamma*DM2c)

d,  v  = LA.eig(L)
dh, vh = LA.eig(Lb.H)

idx = np.argsort(-np.real(d))
d   = d[idx]
v   = v[:,idx]

idx = np.argsort(-np.real(dh))
dh  = dh[idx]
vh  = vh[:,idx]

plt.figure()
plt.axhline(y=0,color='k')
plt.axvline(x=0,color='k')
plt.scatter( np.imag(l),np.real(l),s=40,facecolors='none',edgecolors='k',label='analytical')
plt.scatter(-np.imag(l),np.real(l),s=40,facecolors='none',edgecolors='k')
plt.scatter( np.imag(d),np.real(d),s=20,color='r',label='FD')
plt.scatter( np.imag(dh),np.real(dh),s=20,color='r')
plt.scatter(np.imag(d[0]),np.real(d[0]),s=60,marker='x')
plt.xlabel('Re')
plt.ylabel('Im')
plt.xlim([-2,2])
plt.ylim([-6,2])
plt.legend()

plt.figure(figsize = (10,6))
plt.subplot(1,2,1)
plt.axhline(y=0,color='k')
plt.plot(x,np.real(v[:,0]),'r', label='Re')
plt.plot(x,np.imag(v[:,0]),'b', label='Im')
plt.plot(x,np.abs(v[:,0]),'k' , label='|.|')
plt.plot(x,-np.abs(v[:,0]),'k')
plt.subplot(1,2,2)
plt.axhline(y=0,color='k')
plt.plot(x,np.real(vh[:,0]),'r', label='Re')
plt.plot(x,np.imag(vh[:,0]),'b', label='Im')
plt.plot(x,np.abs(vh[:,0]),'k' , label='|.|')
plt.plot(x,-np.abs(vh[:,0]),'k')
plt.legend()

plt.show()