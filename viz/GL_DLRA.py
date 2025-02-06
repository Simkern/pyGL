import numpy as np
import os, sys, math
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('../.')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

params = {'text.usetex': False,
          'font.size': 8,
          'legend.fontsize': 8,
          'legend.handlelength': 1.,}
plt.rcParams.update(params)

def fsc(number):
    if number == 0:
        return "0.00E+00"
    
    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10 ** exponent)
    
    # Shift decimal point one place left and adjust exponent
    mantissa /= 10
    exponent += 1
    
    # Format the result
    return f"{mantissa:.2f}E{exponent:+03d}"

plt.close("all")

fldr = r'C:\Users\Simon\ENSAM\local\GL_conv'

make_real = False
if_save = True

# Parameters
L  = 50
x0 = -L/2                      # beginning of spatial domain
x1 = L/2                       # end of spatial domain
Nxc = 128
xc = np.linspace(x0, x1, Nxc+2)

# Parameters of the complex Ginzburg-Landau equation
# basic
U   = 2              # convection speed
cu  = 0.2
cd  = -1
mu0 = 0.38
mu2 = -0.01

mu_scal,__,__,__,__   = CGL(mu0,cu,cd,U)
mu,nu,gamma,Umax,mu_t = CGL2(xc,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# input and output parameters
rkb = 1
x_b = -11
s_b = 1
rkc = 1
x_c = x12
s_c = 1
rkX0 = 10

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xc)

# integration weights
dx = np.diff(xc)
wc = np.zeros(Nxc+2)
wc[:Nxc+1] += dx
wc[1:]     += dx

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

if make_real:
    # make real
    Lr = np.real(Lc[1:-1,1:-1])
    Li = np.imag(Lc[1:-1,1:-1])
    L  = np.matrix([[Lr, -Li],[Li, Lr]])
    Nx = 2*Nxc
    w  = np.hstack((wc[1:-1],wc[1:-1]))
    x  = np.hstack((xc[1:-1],xc[1:-1]))
else:
    L = np.matrix(Lc[1:-1,1:-1])
    w = wc[1:-1]
    x = xc[1:-1]
    Nx = Nxc

# weight matrix for convenience
W = np.diag(w)
Winv = np.diag(1.0/w)

# Input & output
B = np.zeros((Nx, rkb))
B[:,0] = np.exp(-((x - x_b)/s_b)**2)
C = np.zeros((rkc, Nx))
C[0,:] = np.exp(-((x - x_c)/s_c)**2)

# plotting preparation
px, py = np.meshgrid(x, x)
box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

# compute controllability gramian

# direct
Qc = B @ B.T @ W
X = linalg.solve_continuous_lyapunov(L, -Qc)

Qo = C.T @ C @ Winv
Y = linalg.solve_continuous_lyapunov(L.T, -Qo)

r = 100
fname = f"data_GL_lyapconv_X_RK_n{Nx:04d}_rk0_10_r{r:03d}.npy"
Xrk = np.load(os.path.join(fldr,fname))
fig3, ax3 = plt.subplots(1,1)
p = ax3.contourf(Xrk, 100, cmap='RdBu_r')
#fig3.colorbar(p, ax = ax3)
#ax3.set_title('Controllability Gramian')


TO = 1
dt = 0.001
for rk0 in [2, 6, 14, 20 ]:
    fname = f"data_GL_lyapconv_XS_n{Nx:04d}_TO{TO:d}_rk0{rk0:02d}_t{fsc(dt)}.npy"
    Sd = np.load(os.path.join(fldr,fname))
    fname = f"data_GL_lyapconv_XU_n{Nx:04d}_TO{TO:d}_rk0{rk0:02d}_t{fsc(dt)}.npy"
    Ud = np.load(os.path.join(fldr,fname))
    Xd = Ud @ Sd @ Ud.T

    fig4, ax4 = plt.subplots(1,1)
    p = ax4.contourf(Xd, 100, cmap='RdBu_r')
    #fig4.colorbar(p, ax = ax4)
    #ax4.set_title('Observability Gramian')

sys.exit()

fig5, ax5 = plt.subplots(1,1)
U, S, VT = linalg.svd(X)
ax5.scatter(range(Nx), S, 50, 'r', marker='x')

fig6, ax6 = plt.subplots(1,1)
U, S, VT = linalg.svd(Y)
ax6.scatter(range(Nx), S, 50, 'b', marker='x')

for ax in [ax5, ax6]:
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlim(1,50)
    ax.set_xlabel(r'$\#$', fontsize=14)
    ax.set_ylabel(r'$\sigma$', fontsize=14)
    ax.set_aspect(0.8)

for ax in [ax3, ax4]:
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w--')
    ax.set_xlim(x0,x1)
    ax.set_ylim(x0,x1)
    ax.axis('off')
    ax.set_ylim(ax.get_ylim()[::-1])    
    
# superimpose BBT and CTC
ax3.scatter(x_b, x_b, 80, 'r', edgecolors='k', linewidths=2)
ax4.scatter(x_c, x_c, 80, 'b', edgecolors='k', linewidths=2)

if if_save:
    names = ['Xctl',
             'Yobs',
             'sigmaX',
             'sigmaY']
    for fig, name in zip([fig3, fig4, fig5, fig6], names):
        fig.savefig(os.path.join('pics_png',name+'.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(os.path.join('pics_eps',name+'.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)