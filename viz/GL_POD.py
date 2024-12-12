import numpy as np
import os
import sys
from time import time as tm

from scipy import linalg
from matplotlib import pyplot as plt

sys.path.append('../.')

from core.CGL_parameters import CGL, CGL2
from core.utils import pmat as pm
from core.utils import pvec as pv
from core.diff_mat import FDmat
from solvers.CN_integrators import CN_L_integrate

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
L0 = 50
x0 = -L0/2                      # beginning of spatial domain
x1 = L0/2                       # end of spatial domain
Nxc = 128
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

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xc)

# integration weights
dx = np.diff(xc)
wc = np.ones(Nxc+2)*dx[0]
w = wc[1:-1]

# linear operator
Lc = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

# Input & output
B = np.zeros((Nx, rkb))
B[:, 0] = np.exp(-((xc - x_b)/s_b)**2)
C = np.zeros((rkc, Nx))
C[0, :] = np.exp(-((xc - x_c)/s_c)**2)

# weight matrix for convenience
W = np.diag(wc)

# direct
Qc = B @ B.T @ W
X = linalg.solve_continuous_lyapunov(Lc, -Qc)

# integration example
q = CN_L_integrate(xc, tvec, mu, nu, gamma, B[:, 0])

# POD
dtau = 1.0 #0.5
step = round(dtau/dt)
qPOD = q[:,::step]

# plotting preparation
pxt, pyt = np.meshgrid(xc, tvec[::step])
pxtt, pytt = np.meshgrid(tvec[::step], tvec[::step])
box = x12*np.array([[1, 1, -1, -1, 1], [1, -1, -1, 1, 1]])
dot = np.array([[x_b, x_c]])

######################################
#
#  DIRECT
#
######################################

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
p = ax1.contourf(pxt, pyt, abs(qPOD).T, 100)

# Add vertical dashed lines at x1 and x2
ax1.axvline(x=x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=-x12, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=x_b, color='red', linestyle=':', linewidth=4)

# Set axis labels using TeX
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

fig2, axs = plt.subplots(1, 2, figsize=(10, 8))
p0 = axs[0].contourf(pxt, pyt, np.real(qPOD).T, 100)
p1 = axs[1].contourf(pxt, pyt, np.imag(qPOD).T, 100)

for ax in axs:
    # Add vertical dashed lines at x1 and x2
    ax.axvline(x=x12, color='white', linestyle='--', linewidth=1)
    ax.axvline(x=-x12, color='white', linestyle='--', linewidth=1)
    ax.axvline(x=x_b, color='red', linestyle=':', linewidth=4)

    # Set axis labels using TeX
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim([x0, x1])
    ax.set_ylim([0, T])

nsnap = qPOD.shape[1]
colors = plt.cm.jet(np.linspace(0,1,nsnap))
fig1, axs = plt.subplots(1, 2, figsize=(10, 8))
for i in range(nsnap):
    snap = qPOD[:, i]  # Get the column by indexing
    axs[0].plot(xc, np.real(snap), color=colors[i])
    axs[1].plot(xc, np.imag(snap), color=colors[i])


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
dtauv = [ 0.1, 0.5, 1.0, 2.0 ]
colors = plt.cm.jet(np.linspace(0,1,len(dtauv)))
nplot = 30
ssvd = linalg.svd(X, compute_uv=False)
ssvdpt = np.vstack((ssvd,ssvd)).T.flatten()[:2*nplot]
ax.scatter(range(1,2*nplot+1), ssvdpt, 50, 'k', marker='o', label='exact')

for i, dtau in enumerate(dtauv[::-1]):
    step = round(dtau/dt)
    qPOD = q[:,::step]
    qr = np.real(qPOD)
    qi = np.imag(qPOD)
    Nx, Nt = qPOD.shape
    if Nx > Nt:
        XHX = qPOD.conj().T @ W @ qPOD * T/Nt
    else:
        XHX = qPOD @ qPOD.conj().T @ W * T/Nt

    s = linalg.svdvals(XHX)
    npt = min(s.size, nplot)
    spt = np.vstack((s[:npt],s[:npt])).T.flatten()
    ax.scatter(range(1,2*npt+1), spt, 50, marker='x', label=f'dt = {dtau:.5f}')
ax.set_ylim(1e-12, 1e4)
ax.set_xlim(0,2*npt)
ax.set_yscale('log')
plt.legend()
plt.show()
