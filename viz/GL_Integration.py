import numpy as np
import os
import sys
from time import time as tm

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

sys.path.append('../.')

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat
from core.utils import enorm
from solvers.CN_integrators import CN_Lf_integrate, CN_Lf_adj_integrate

params = {'text.usetex': False,
          'font.size': 25,
          'axes.labelsize' : 20, 
          'xtick.labelsize': 14,  # X-tick label font size
          'ytick.labelsize': 20,  # Y-tick label font size
          'legend.fontsize': 16,
          'legend.handlelength': 1.,}
plt.rcParams.update(params)

plt.close("all")

make_real = False
if_save = False

# Parameters
L = 50
x0 = -L/2                      # beginning of spatial domain
x1 = L/2                       # end of spatial domain
Nxc = 256
xc = np.linspace(x0, x1, Nxc+2)
Nx = len(xc)
T = 100
dt = 0.1
Nt = int(T / dt)                # number of timesteps
tvec = np.linspace(0, T, Nt)       # temporal grid

# Parameters of the complex Ginzburg-Landau equation
# basic
U = 2              # convection speed
cu = 0.2
cd = -1
mu0 = 0.38
mu2 = -0.01

mu_scal, __, __, __, __ = CGL(mu0, cu, cd, U)
mu, nu, gamma, Umax, mu_t = CGL2(xc, mu0, cu, cd, U, mu2, True)
x12 = np.sqrt(-2*mu_scal/mu2)

# input and output parameters
rkb = 1
x_b = -11
s_b = 1
rkc = 1
x_c = x12
s_c = 1
rkX0 = 10

# Input & output
B = np.zeros((Nx, rkb))
B[:, 0] = np.exp(-((xc - x_b)/s_b)**2)
C = np.zeros((rkc, Nx))
C[0, :] = np.exp(-((xc - x_c)/s_c)**2)

# plotting preparation
pxt, pyt = np.meshgrid(xc, tvec)
box = x12*np.array([[1, 1, -1, -1, 1], [1, -1, -1, 1, 1]])
dot = np.array([[x_b, x_c]])

# integration example
rtime = np.random.randn(Nt,)
fq = np.outer(B[:, 0], rtime)
q = CN_Lf_integrate(xc, tvec, mu, nu, gamma, np.zeros((Nx,)), fq)

rtime = np.random.randn(Nt,)
fpsi = np.outer(C[0,:].T, rtime)
psi = CN_Lf_adj_integrate(xc, tvec, mu, nu, gamma, np.zeros((Nx,)), fpsi)

######################################
#
#  DIRECT
#
######################################

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
p = ax1.contourf(pxt, pyt, abs(q).T, 100)
#ax1.set_title('GL time-integration')

# Add vertical dashed lines at x1 and x2
ax1.axvline(x=x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=-x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=x_b, color='red', linestyle=':', linewidth=4)

# Set axis labels using TeX
#ax1.set_xlabel(r'$x$', fontsize=14)
#ax1.set_ylabel(r'$t$', fontsize=14)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim([x0, x1])
ax1.set_ylim([0, T])

# Customize colorbar
cbar = fig1.colorbar(p, ax=ax1, 
                     orientation = 'horizontal', 
                     location    = 'top',
                     shrink      = 0.8, 
                     aspect      = 30)
cbar.ax.tick_params(left=False, right=False, top=False, bottom=False)  # Remove tick marks
cbar.ax.set_xticklabels([])  # Remove numerical ticks
cbar.set_label(r'$|q(x,t)|$', fontsize=20, labelpad=10)

# Set plot title
#ax1.set_title('GL Time-Integration', fontsize=16)

#####################################################

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 2))
# Second subplot - Plot of B[:,0] over xc
ax2.plot(xc, B[:, 0], color='red', label=r'$B[:,0]$', linewidth=2)
# ax2.yaxis.set_visible(False)
ax2.set_xlabel(r'$x$', fontsize=20)
ax2.set_ylabel(r'$B$', fontsize=20)
ax2.axvline(x=x12, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=-x12, color='black', linestyle='--', linewidth=1)
ax2.set_xlim([x0, x1])
#ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelleft=False)
ax2.tick_params(axis='y', which='both', left=False,right=False, labelleft=False)
ax2.set_aspect(aspect=5.0)

fig3, ax3 = plt.subplots(1, 1, figsize=(2, 4))
# Second subplot - Plot of rtime over tvec
ax3.plot(rtime, tvec, color='red', label=r'$f$', linewidth=1)
#ax2.set_xlabel(r'$x$', fontsize=14)
ax3.set_ylabel(r'$t$', fontsize=14)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#ax3.tick_params(axis='y', which='both', left=False,   right=False, labelleft=False)
ax3.set_ylim([0, T])
ax3.set_aspect(aspect=1)


######################################
#
#  ADJOINT
#
######################################

fig4, ax1 = plt.subplots(1, 1, figsize=(10, 8))
p = ax1.contourf(pxt, pyt, abs(psi).T, 100)
#ax1.set_title('GL time-integration')

# Add vertical dashed lines at x1 and x2
ax1.axvline(x=x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=-x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=x_c, color='blue', linestyle=':', linewidth=4)

# Set axis labels using TeX
#ax1.set_xlabel(r'$x$', fontsize=14)
#ax1.set_ylabel(r'$t$', fontsize=14)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim([x0, x1])
ax1.set_ylim([0, T])

# Customize colorbar
cbar = fig1.colorbar(p, ax=ax1, 
                     orientation = 'horizontal', 
                     location    = 'top',
                     shrink      = 0.8, 
                     aspect      = 30)
cbar.ax.tick_params(left=False, right=False, top=False, bottom=False)  # Remove tick marks
cbar.ax.set_xticklabels([])  # Remove numerical ticks
cbar.set_label(r'$|\psi(x,t)|$', fontsize=20, labelpad=10)

# Set plot title
#ax1.set_title('GL Time-Integration', fontsize=16)

#####################################################

fig5, ax2 = plt.subplots(1, 1, figsize=(10, 4))
# Second subplot - Plot of B[:,0] over xc
ax2.plot(xc, C[0, :], color='blue', label=r'$C[0,:]$', linewidth=2)
# ax2.yaxis.set_visible(False)
ax2.set_xlabel(r'$x$', fontsize=14)
ax2.set_ylabel(r'$B$', fontsize=14)
ax2.axvline(x=x12, color='black', linestyle='--', linewidth=1)
ax2.axvline(x=-x12, color='black', linestyle='--', linewidth=1)
ax2.set_xlim([x0, x1])
#ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelleft=False)
ax2.tick_params(axis='y', which='both', left=False,right=False, labelleft=False)
ax2.set_aspect(aspect=10.0)

fig6, ax3 = plt.subplots(1, 1, figsize=(2, 4))
# Second subplot - Plot of rtime over tvec
ax3.plot(rtime, tvec, color='blue', label=r'$f$', linewidth=1)
#ax2.set_xlabel(r'$x$', fontsize=14)
ax3.set_ylabel(r'$t$', fontsize=14)
ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#ax3.tick_params(axis='y', which='both', left=False,   right=False, labelleft=False)
ax3.set_ylim([0, T])
ax3.set_aspect(aspect=1)

names = ['CGL_direct',
         'B',
         'CGL_direct_forcing',
         'CGL_adjoint',
         'CT',
         'CGL_adjoint_forcing']

if if_save:
    for fig, name in zip([fig1, fig2, fig3, fig4, fig5, fig6], names):
        fig.savefig(os.path.join('pics_png',name+'.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(os.path.join('pics_eps',name+'.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
else:
    plt.show()
