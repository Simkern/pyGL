import numpy as np

from matplotlib import pyplot as plt
#from matplotlib import colors
from matplotlib import cm

from diff_mat import *
from CGL_parameters import *
from integrators import *

# Parameters
x0 = 0                        # beginning of spatial domain
x1 = 30                         # end of spatial domain
dx = 0.1                        # Spatial discretisation
dt = 0.01                       # Time step
T  = 10                         # Total simulation time

# Discretisation grids
L  = x1-x0                      # spatial domain size
Nx = int(L / dx)                # number of spatial dof
Nt = int(T / dt)                # number of timesteps
x = np.linspace(x0, x1, Nx)     # spatial grid
t = np.linspace(0, T, Nt)       # temporal grid

# Boundary conditions (Dirichlet 0)
tmp = np.ones(Nx,)
tmp[1:-1] = 0
bcmsk = np.array(tmp,dtype='bool')
msk   = np.invert(bcmsk)

# Initial condition
# gaussian pulse centered at m0 with standard deviation s0
m0 = 5
s0 = 0.5
q0 = np.exp(-((x-m0)/s0)**2)
q0[bcmsk] = 0

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.1
cd  = -1
mu0 = 0.01
mu,nu,gamma,Umax,mu_t = CGL(mu0,cu,cd,U,True)

mu0v = [0, 0.21] #[ 0, 0.06, 0.09 ]
Nm = len(mu0v)

print(f'Cases:\n')
for im,mu0 in enumerate(mu0v):
    # update mu
    mu, __, __, __, __ = CGL(mu0,cu,cu,U)
    print(f'  {im+1}: mu_0 = {mu0:.2f}  -->  mu = {mu:.2f}')
print('\n')

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(x[1:-1])
I = sp.eye(Nx-2)

# Intialise fields
q = np.zeros((Nx,Nt,Nm), dtype=complex)
for im in range(Nm):
    q[:,0,im] = q0

print(f'\nLinear simulations:')
for im,mu0 in enumerate(mu0v):
    # update mu
    mu, __, __, __, __ = CGL(mu0,cu,cu,U)
    print(f'Case {im+1}: mu = {mu:.2f}')
    L = mu*I - nu*DM1b + gamma*DM2c
    for it in range(Nt-1):
        q[msk,it+1,im] = CN_L_advance(q[msk,it,im],L,dt)

# Plot the solution
nl = 5
tt,xx = np.meshgrid(t,x)
idx = np.argmin(abs(x-m0))

plt.figure(figsize=(10, 10))
for im in range(Nm):
    i = 3*im + 1
    qp = np.real(q[:,:,im])
    plt.subplot(Nm,3,i)
    for it in range(0,Nt,int(np.floor(Nt/nl))):
        plt.plot(x,qp[:,it],color='k',alpha=(3/4-it/(2*Nt)))
    plt.plot(x,qp[:,0], label='IC', color='r')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend
    
    i = i+1
    plt.subplot(Nm,3,i)
    plt.set_cmap('bwr')
    vmi = -1 #qp.min()
    vma =  2 #qp.max()
    #norm = colors.TwoSlopeNorm(vmin=vmi, vcenter=0, vmax=vma)
    plt.pcolor(xx,tt,qp,vmin=vmi, vmax=vma)
    plt.plot(np.zeros((Nt,1)),t,'k--')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    i = i+1
    plt.subplot(Nm,3,i)
    plt.plot(t,qp[idx,:], color='k')
    plt.xlabel('t')
    plt.ylabel('Amplitude')

plt.show()