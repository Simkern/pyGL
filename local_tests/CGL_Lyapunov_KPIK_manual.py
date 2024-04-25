import numpy as np
import sys
import time

from core.utils import pvec, pmat

from scipy import linalg
from matplotlib import pyplot as plt

from core.CGL_parameters import CGL, CGL2
from core.diff_mat import FDmat

from solvers.arnoldi import arn, arn_inv
from solvers.lyapunov import lrcfadic_r

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
mu,nu,gamma,Umax,mu_t = CGL2(xvec,mu0,cu,cd,U,mu2,True)
x12 = np.sqrt(-2*mu_scal/mu2)

# Generate differentiation matrices
DM1f,DM1b,DM2c = FDmat(xvec)

A = np.asarray(np.diag(mu) - nu*DM1f + gamma*DM2c)

Ar = np.real(A)
Ai = np.imag(A)

# make real
A = np.block([[Ar, -Ai],[Ai, Ar]])
Nx = 2*Nxc

# rhs
m = 1
B = np.zeros((Nx,m))
B[-1] = 400

'''
B = np.array([[1,0,0,0,0,0,0,0,0,1],
              [0,1,0,0,0,0,0,1,1,1]]).T

A = np.array([[2, 8, 4, 3, 5, 1, 7, 8, 8, 6],
              [8, 1, 7, 5, 5, 3, 8, 7, 3, 8],
              [8, 3, 9, 4, 2, 5, 7, 4, 4, 8],
              [4, 8, 5, 6, 2, 4, 9, 5, 1, 7],
              [8, 2, 4, 7, 4, 7, 8, 2, 6, 4],
              [8, 5, 8, 4, 7, 9, 1, 6, 1, 2],
              [7, 8, 3, 3, 4, 2, 4, 5, 3, 9],
              [6, 9, 6, 7, 8, 9, 5, 1, 1, 7],
              [6, 1, 9, 7, 2, 5, 9, 9, 5, 9],
              [4, 6, 7, 3, 7, 9, 5, 1, 7, 2]])
'''
n, sh = B.shape
s     = 2*sh

# compute LU factorisation of A
luA,piv = linalg.lu_factor(A)

# initialise Krylov basis
V1 = np.column_stack([ B , linalg.lu_solve((luA,piv),B ) ])
# Orthogonalize
U,beta   = linalg.qr(V1,mode='economic')
beta_inv = linalg.inv(beta)

beta  = beta[:sh,:sh]
beta2 = beta @ beta.T

k_max = 50

normB = np.linalg.norm(B, 'fro')**2
normA = np.linalg.norm(A, 'fro')
sqrt2 = np.sqrt(2)
error2 = np.zeros(k_max)

odds = []

s = 2 * sh
B1 = linalg.lu_solve((luA,piv), B)

H    = np.zeros(((k_max+1)*s,k_max*s))
T    = np.zeros(((k_max+1)*s,k_max*s))
L    = np.zeros(((k_max+1)*s,k_max*s))

tol = 1e-12
tolY = 1e-12
rho = 1 #dummy

etime = time.time()
  
for j in range(k_max):
    
    # indices
    jms = j*s
    j1s = (j + 2) * s
    js  = (j + 1) * s
    js1 = js
    jsh = j*s + sh
    
    # compute new vectors for the Krylov basis
    Up = np.zeros((n, s))
    Up[:, :sh] = A @ U[:, jms:jsh]
    Up[:, sh:] = linalg.lu_solve((luA,piv), U[:, jsh:js])
    
    # orthogonalise new vector wrt. current Krylov basis, add to Hessenberg matrix
    for l in range(2):      # MGS
        k_min = max(0, j - k_max - 1)
        for kk in range(k_min, j + 1):
            k1 = kk * s
            k2 = (kk+1) * s
            proj              = U[:, k1:k2].T @ Up
            H[k1:k2, jms:js] += proj
            Up               -= U[:, k1:k2] @ proj
            #pmat(H)


    # orthogonalise new vectors wrt. to each other, add subdiagonal block to Hessenberg matrix
    if j < k_max:
        Up, H[js1:j1s, jms:js] = linalg.qr(Up, mode='economic')
        H_inv                  = np.linalg.inv(H[js1:j1s, jms:js])

    # determine the coefficient matrix for the projected problem
    # NOTE: avoids the explicit multiplication with A
    I = np.eye(js + s)
    if j == 0:
        # A/B = (B.T\A.T).T
        V1 = linalg.solve(beta_inv[:sh, :sh].T,   H[:s+sh, :sh].T).T
        V2 = linalg.solve(beta_inv[:sh, :sh].T,np.eye(s+sh, sh).T).T
        L[:s+sh,:sh] = np.block([ V1, V2 ]) @ beta_inv[:s, sh:s]
    else:
        L[:js+s,j*sh:(j+1)*sh] += H[:js+s, jms:jms+sh] @ rho
        
    odds.extend(list(range(jms, jms + sh)))
    evens = list(range(js))
    evens = [x for x in evens if x not in odds]
    
    
    # odd columns
    T[:js+s,  odds]  = H[:js+s,  odds]
    # even columns
    T[:js+sh, evens] = L[:js+sh, :(j+1)*sh]

    L[:js+s, (j+1)*sh:(j+2)*sh] = ( I[:js+s, js-sh:js] 
                               - T[:js+s, :js] @ H[:js, js-sh:js] 
                               ) @ H_inv[sh:s, sh:s]

    rho = np.linalg.inv(H_inv[:sh, :sh]) @ H_inv[:sh, sh:s]

    # Solve the projected problem using Stuart-Bartels
    Y = linalg.solve_lyapunov(T[:js, :js], np.eye(js, sh) @ beta2 @ np.eye(sh,js))
    # Ensure that the result is symmetric
    Y = (Y + Y.T) / 2

    # Compute residual
    cc = np.block([H[js1:j1s, js-s:js-sh], L[js1:j1s, j*sh:(j+1)*sh]])
        
    normX = np.linalg.norm(Y, 'fro')
    error2[j] = sqrt2 * np.linalg.norm(cc @ Y[js-s:js, :], 'fro') / (normB + normA * normX)

    #print([j + 1, error2[j]])

    if error2[j] < tol:
        break
    else:
        U = np.block([U,Up])

uY, sY, _ = np.linalg.svd(Y)
is_ = np.sum(np.abs(sY) > tolY)
Y0 = uY[:, :is_] @ np.diag(np.sqrt(sY[:is_]))

Z = U[:, :js] @ Y0
error2 = error2[:j + 1]
etime = time.time() - etime

print(' its |        comp.res. | space dim. |       CPU Time')
print(f'{j + 1:4d} |   {error2[j]:14.12f} |       {js:4d} | {etime:14.12f}')
   
                     
