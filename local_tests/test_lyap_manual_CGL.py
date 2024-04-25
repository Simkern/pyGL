import numpy as np
import time

from scipy import linalg
import sys

from git.CGL_parameters import CGL, CGL2
from git.diff_mat import FDmat
import git.utils
from matplotlib import pyplot as plt

import arnoldi
import lyap

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
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,False)
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
na = 10
__,Ha = arnoldi.arn(A,b0,na)
pA,__ = linalg.eig(Ha[0:na,0:na])
pA0    = sorted(pA)
# Compute the set of Ritz values R- for A^-1
nb = 10
__,Hb    = arnoldi.arn_inv(A,b0,nb)
Dbtmp,__ = linalg.eig(Hb[0:nb,0:nb])
pAinv0 = np.array([ 1/r for r in sorted(Dbtmp) ])

pA = np.empty((2*na,), dtype = complex)
for i in range(na):
    pA[2*i]   = pA0[i]
    pA[2*i+1] = pA0[i].conj()
pAinv = np.empty((2*nb,), dtype = complex)
for i in range(nb):
    pAinv[2*i]   = pAinv0[i]
    pAinv[2*i+1] = pAinv0[i].conj()
    
m = 1

I = np.eye(Nx)

B = np.zeros((Nx,m))
B[-1] = 400

Xref = linalg.solve_continuous_lyapunov(A, -B @ B.T)
nrmx0 = linalg.norm(Xref, 'fro')

print(np.allclose(A @ Xref + Xref @ A.T, -B @ B.T))
#print(nrmx0)

print('n = 30, pA,pAinv = (30,0)')
pin = pA
idx = np.argsort(-np.abs(pin))
pin = pin[idx]
p_v = np.array([])

l = pin.size

eps = 1e-12
for i in range(l):
    if np.abs(np.imag(pin[i])) < eps:
        pin[i] = np.real(pin[i])

for (i,p) in enumerate(pin[:-1]):
    if not np.abs(p - pin[i+1]) < eps and not p.conj() == pin[i+1]:
        p_v = np.append(p_v,p)
        
if np.imag(pin[-1]) == 0:
    p_v = np.append(p_v,pin[-1])
    
l = p_v.size
    
I = np.eye(A.shape[0])

lu_rl = []
lu_cc = []
is_real = np.imag(p_v) == 0

for i, (shift, is_r) in enumerate(zip(p_v,is_real)):
    if is_r:
        lu, piv = linalg.lu_factor(A + shift*I)
        lu_rl.append((lu,piv))
        lu_cc.append((0,0))
    else:
        s_i = 2*np.real(-shift)
        t_i = np.abs(shift)**2
        lu, piv = linalg.lu_factor(A @ A + s_i*A + t_i*I)
        lu_rl.append((0,0))
        lu_cc.append((lu,piv))
         
# for i in range(l):
#     print(f'p[{i}] = {p_v[i]}')
#     if is_real[i]:
#         if not lu_cc[i][0] == 0:
#             print('error')
#         else:
#             print('  real: ok!')
#     else:
#         if not lu_rl[i][0] == 0:
#             print('error')
#         else:
#             print('  complex: ok!')
          
#print(p_v)
#p_v = np.flip(p_v)
#lu_rl.reverse()
#lu_cc.reverse()
print(p_v)
p_v = p_v[:1]
l = p_v.size

print(linalg.norm(Xref, 'fro')/nrmx0)

## Start loop
p_old = 1

i  = 0
ip = 0

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

q  = np.sqrt(-2*p)
if is_real:
    print(f'i = {i}: p     = {np.real(p):.2f} :                     real')
    #lu, piv = linalg.lu_factor(A + p*I)
    #V = q*linalg.lu_solve((lu, piv), B)
    Vb = q*linalg.lu_solve(lu_rl[ip], B)
else:
    print(f'i = {i}: p     = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
    q2 = 2*np.sqrt(-np.real(p))
    
    #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
    #V1 = q1*linalg.lu_solve((lu,piv), B)
    V1 = q1*linalg.lu_solve(lu_cc[ip], B)
    V2 = q2*(A @ V1)
    Vb  = np.column_stack([V1, V2])
Z = Vb
    
p_old = p
i = i + 1
ip = i % l       # update shift index   

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)

sys.exit()

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

if is_real:
    print(    f'i = {i}: p     = {np.real(p):.2f} :                     real')
    if is_real_old:
        print(f'       p_old = {np.real(p_old):.2f} :                     real')
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V - (p + p_old)*linalg.lu_solve((lu,piv),V)
        Vnew = Vb - (p + p_old)*linalg.lu_solve(lu_rl[ip],Vb)
    else:
        print(f'         p_old = {np.real(p_old):.2f} + i {np.imag(p_old):.2f}: complex conjugate')
        q1 = 2*np.real(p_old + p)
        q2 = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve((lu,piv),V1)
        Vnew = V2 - q1*V1 + q2*linalg.lu_solve(lu_rl[ip],V1)
    V = q*Vnew
    Vb = V
else:
    print(f'i = {i}: p = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    if is_real_old:
        Vt   = A @ V - p_old*V
        
        lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        Vnew = linalg.lu_solve((lu,piv),Vt)
        #Vnew = linalg.lu_solve(lu_cc[ip],Vt)
    else:
        q1   = np.abs(p_old)**2 - np.abs(p)**2
        q2   = 2*np.real(p_old + p)
        Vt   = q1*V1 - q2*V2
        
        lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        Vnew = V1 + linalg.lu_solve((lu,piv), Vt)
        #Vnew = V1 + linalg.lu_solve(lu_cc[ip], Vt)
    V1 = q*np.abs(p)*Vnew
    V2 = q*(A @ Vnew)
    Vb = np.column_stack([ V1, V2 ])  
Z = np.column_stack([ Z, Vb ])

p_old = p
i = i + 1
ip = i % l       # update shift index   

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

if is_real:
    print(    f'i = {i}: p     = {np.real(p):.2f} :                     real')
    if is_real_old:
        print(f'       p_old = {np.real(p_old):.2f} :                     real')
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V - (p + p_old)*linalg.lu_solve((lu,piv),V)
        Vnew = Vb - (p + p_old)*linalg.lu_solve(lu_rl[ip],Vb)
    else:
        print(f'         p_old = {np.real(p_old):.2f} + i {np.imag(p_old):.2f}: complex conjugate')
        q1 = 2*np.real(p_old + p)
        q2 = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve((lu,piv),V1)
        Vnew = V2 - q1*V1 + q2*linalg.lu_solve(lu_rl[ip],V1)
    V = q*Vnew
    Vb = V
else:
    print(f'i = {i}: p = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    if is_real_old:
        Vt   = A @ V - p_old*V
        
        lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        Vnew = linalg.lu_solve((lu,piv),Vt)
        #Vnew = linalg.lu_solve(lu_cc[ip],Vt)
    else:
        q1   = np.abs(p_old)**2 - np.abs(p)**2
        q2   = 2*np.real(p_old + p)
        Vt   = q1*V1 - q2*V2
        
        lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        Vnew = V1 + linalg.lu_solve((lu,piv), Vt)
        #Vnew = V1 + linalg.lu_solve(lu_cc[ip], Vt)
    V1 = q*np.abs(p)*Vnew
    V2 = q*(A @ Vnew)
    Vb = np.column_stack([ V1, V2 ])  
Z = np.column_stack([ Z, Vb ])

p_old = p
i = i + 1
ip = i % l       # update shift index   

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)