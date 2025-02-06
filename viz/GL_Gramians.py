import numpy as np
import os, sys
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('../.')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

params = {'text.usetex': False,
          'font.size': 25,
          'axes.labelsize' : 16, 
          'xtick.labelsize': 12,  # X-tick label font size
          'ytick.labelsize': 12,  # Y-tick label font size
          'legend.fontsize': 14,
          'legend.handlelength': 1.,}
plt.rcParams.update(params)

plt.close("all")

make_real = True
if_save = False

# Parameters
L0  = 50
x0 = -L0/2                      # beginning of spatial domain
x1 = L0/2                       # end of spatial domain
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
mu,nu,gamma,Umax,mu_t = CGL2(xc,mu0,cu,cd,U,mu2,False)
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
wc = np.ones(Nxc+2)*dx[0]

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)
# Input & output
B = np.zeros((Nxc+2, rkb))
B[:,0] = np.exp(-((xc - x_b)/s_b)**2)
C = np.zeros((rkc, Nxc+2))
C[0,:] = np.exp(-((xc - x_c)/s_c)**2)

if make_real:
    # make real
    Lr = np.real(Lc[1:-1,1:-1])
    Li = np.imag(Lc[1:-1,1:-1])
    L  = np.block([[Lr, -Li],[Li, Lr]])
    Nx = 2*Nxc
    w  = np.hstack((wc[1:-1],wc[1:-1]))
    x  = np.hstack((xc[1:-1],xc[1:-1]))
    # Input & output
    Br = np.real(B[1:-1,:])
    Bi = np.imag(B[1:-1,:])
    B = np.block([[Br, -Bi], [Bi, Br]])
    Cr = np.real(C[:,1:-1])
    Ci = np.imag(C[:,:1:-1])
    C = np.block([[Cr, -Ci], [Ci, Cr]])
    # plotting preparation
    xp = np.hstack((xc[1:-1],xc[1:-1] + L0))
    px, py = np.meshgrid(xp, xp)
else:
    L = np.matrix(Lc[1:-1,1:-1])
    w = wc[1:-1]
    x = xc[1:-1]
    Nx = Nxc
    # Input & Output
    B = B[1:-1,:]
    C = C[:,1:-1]
    # plotting preparation
    xp = x
    px, py = np.meshgrid(x, x)

# weight matrix for convenience
W    = np.diag(w)
Winv = np.diag(1.0/w)

box = x12*np.array([[1,1,-1,-1,1],[1,-1,-1,1,1]])
dot = np.array([[x_b, x_c]])

# compute controllability gramian

# direct
Qc = B @ B.T @ W
X = linalg.solve_continuous_lyapunov(L, -Qc)

Qo = C.T @ C @ Winv
Y = linalg.solve_continuous_lyapunov(L.T, -Qo)

fig1, ax1 = plt.subplots(1,1)
p = ax1.contourf(px, py, Qc, 100)
#fig1.colorbar(p, ax = ax1)
ax1.set_title('B @ B.T', fontsize=14)

fig2, ax2 = plt.subplots(1,1)
p = ax2.contourf(px, py, Qo, 100)
#fig2.colorbar(p, ax = ax2)
ax2.set_title('C.T @ C', fontsize=14)

fig3, ax3 = plt.subplots(1,1)
p = ax3.contourf(px, py, X, 100, cmap='RdBu_r')
#fig3.colorbar(p, ax = ax3)
ax3.set_title('Controllability Gramian', fontsize=14)

fig4, ax4 = plt.subplots(1,1)
p = ax4.contourf(px, py, Y, 100, cmap='RdBu_r')
#fig4.colorbar(p, ax = ax4)
ax4.set_title('Observability Gramian', fontsize=14)

fig5, ax5 = plt.subplots(1,1)
ax5.scatter(range(1,Nx+1), linalg.svd(X, compute_uv=False), 50, 'k', marker='+', label='exact')
ax5.set_title('Controllability Gramian', fontsize=14)

fig6, ax6 = plt.subplots(1,1)
ax6.scatter(range(1,Nx+1), linalg.svd(Y, compute_uv=False), 50, 'k', marker='+', label='exact')
ax6.set_title('Observability Gramian', fontsize=14)

for ax in [ax5, ax6]:
    ax.set_yscale('log')
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlim(0,60)
    ax.set_xlabel(r'$\#$')
    ax.set_ylabel(r'$\sigma$')
    #ax.set_aspect(3)
    ax.legend()

for ax in [ax1, ax2, ax3, ax4]:
    ax.axis('equal') 
    ax.plot(box[0,:],box[1,:],'w--')
    if make_real:
        ax.plot(box[0,:]+L0,box[1,:],   'w--')
        ax.plot(box[0,:],   box[1,:]+L0,'w--')
        ax.plot(box[0,:]+L0,box[1,:]+L0,'w--')
    ax.set_xlim(min(xp), max(xp))
    ax.set_ylim(min(xp), max(xp))
    ax.axis('off')
    #ax.set_ylim(ax.get_ylim()[::-1])    
for fig in [ fig5, fig6 ]:
    fig.set_figheight(5)
    fig.set_figwidth(4)
# superimpose BBT and CTC
ax3.scatter(x_b, x_b, 80, 'r', edgecolors='k', linewidths=2)
ax4.scatter(x_c, x_c, 80, 'b', edgecolors='k', linewidths=2)
if make_real:
    ax3.scatter(x_b + L0, x_b + L0, 80, 'r', edgecolors='k', linewidths=2)
    ax4.scatter(x_c + L0, x_c + L0, 80, 'b', edgecolors='k', linewidths=2)

if if_save:
    names = ['BBT',
             'CTC',
             'Xctl',
             'Yobs',
             'sigmaX',
             'sigmaY']
    if make_real:
        for i in range(len(names)):
            names[i] += '_real'
    for fig, name in zip([fig1, fig2, fig3, fig4, fig5, fig6], names):
        fig.savefig(os.path.join('pics_png',name+'.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(os.path.join('pics_eps',name+'.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
else:
    plt.show()