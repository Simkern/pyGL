import numpy as np
import sys
import copy as cp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, gmres
from scipy.linalg import orth

N = 50

tolgmres = 1e-6
tolcg = 1e-6
amp = 5
restart = 90

'''
l = list(np.linspace(1,20,N))

V0 = orth(np.random.rand(N,N))
L = np.diag(l)

A = V0 @ L @ V0.T

op = 0.5*(A + A.T)

E0,V = np.linalg.eig(op)

P = cp.copy(op)

#op += np.diag(np.random.rand(N,1))*1
op += np.random.rand(N,N)*amp

E,V = np.linalg.eig(op)

#print(l)
#print(np.sort(E))

E1,V1 = np.linalg.eig(np.linalg.solve(P,op))

plt.figure()
plt.scatter(np.real(E),np.imag(E),label='E')
plt.scatter(np.real(E0),np.imag(E0),label='E0')
plt.scatter(np.real(E1),np.imag(E1),label='E1')
plt.legend()

#sys.exit()
'''

rho= np.sqrt(np.random.rand(N)*10)
th = np.random.rand(N)*2*np.pi
d = rho*np.exp(1j*th) + amp
dd = np.diag(np.linspace(1,10,N))

V = orth(np.random.rand(N,N))
op = np.linalg.inv(V) @ np.diag(d) @ V + dd
op = (op + op.conj().T)/2

A = cp.copy(op)

D,X = np.linalg.eig(A)

b = np.ones((N,))

P = np.diag(np.diagonal(A))

PE,XE = np.linalg.eig(P)

DE,XE = np.linalg.eig(np.linalg.solve(P,A))

plt.figure()
plt.scatter(np.real(D),np.imag(D),label='D')
plt.scatter(np.real(PE),np.imag(PE),label='P')
plt.scatter(np.real(DE),np.imag(DE),label='Pinv A)')

cgres = []

def cg_callback(xk):
    cgres.append(np.linalg.norm(P @ xk - b))

b = np.ones((N,1))
x, info = cg(P,b,tol=tolcg,callback = cg_callback)

xref = np.squeeze(np.linalg.solve(P,b))

#print(np.linalg.norm(x - xref))

gmresres = []
gmresrk  = []

def gmres_callback(xk):
    gmresres.append(np.linalg.norm(op @ xk - b))
    
def gmres_callback2(rk):
    gmresrk.append(rk)
    
def cg_prec(v):
    x, info = cg(P,v,tol=tolcg)
    return x

def solve_prec(v):
    x = np.linalg.solve(P,v)
    return x

x, info = gmres(op,b, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback,
                callback_type='x')

x, info = gmres(op,b, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback2,
                callback_type='pr_norm')

prec = LinearOperator((N, N), matvec=cg_prec)
sprec = LinearOperator((N, N), matvec=solve_prec)

gcold = cp.copy(gmresres)
gcold2 = cp.copy(gmresrk)

gmresrk = []
gmresres = []

x, info = gmres(op,b, M=prec, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback,
                callback_type='x')

x, info = gmres(op,b, M=prec, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback2,
                callback_type='pr_norm')

gcprec = cp.copy(gmresres)
gcprec2 = cp.copy(gmresrk)

gmresrk = []
gmresres = []

x, info = gmres(op,b, M=sprec, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback,
                callback_type='x')

x, info = gmres(op,b, M=sprec, tol=tolgmres,
                restart=restart, maxiter=N, 
                callback=gmres_callback2,
                callback_type='pr_norm')

plt.figure()
plt.subplot(121)
plt.axhline(tolgmres,color='k',linestyle='--')
plt.semilogy(gcold,label='GMRES error')
plt.semilogy(gcprec,label='PGMRES (CG) error')
plt.semilogy(gmresres,label='PGMRES (solve) error')
plt.legend()
plt.subplot(122)
plt.semilogy(cgres,label='CG')
plt.axhline(tolgmres,color='k',linestyle='--')
plt.semilogy(gcprec2,label='GMRES res norm')
plt.semilogy(gcold2,label='PGMRES (CG) res norm')
plt.semilogy(gmresrk,label='PGMRES (solve) res norm')
plt.legend()
#print(res)