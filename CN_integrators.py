import scipy.sparse as sp
import numpy as np

from git.diff_mat import FDmat

## Crank-Nicolson
# a = I - 0.5*dt*L
# b = (I + 0.5*dt*L)q + NL(q) + f

def CN_L_integrate(xvec,tvec,mu,nu,gamma,q0):
    
    Nt = len(tvec)
    Nx = len(xvec)
    dt = np.diff(tvec)[0]
    
    DM1f,__,DM2c = FDmat(xvec)

    if len(mu) == 1:
        print('expand mu')
        mu = np.ones((Nx,))*mu    # if mu is scalar

    L = sp.lil_matrix(np.diag(mu) - nu*DM1f + gamma*DM2c)
    
    q   = np.zeros((Nx,Nt), dtype=complex)
    q[:,0] = q0
    
    for it in range(Nt-1):
        q[:,it+1] = CN_L_advance(q[:,it],L,dt)
        
    return q

def CN_L_adj_integrate(xvec,tvec,mu,nu,gamma,psiT,q):
    
    (Nx,Nt) = q.shape
    dt = np.diff(tvec)[0]
    
    __,DM1b,DM2c = FDmat(xvec)

    if len(mu) == 1:
        mu = np.ones((Nx,))*mu    # if mu is scalar

    LH = sp.lil_matrix(np.diag(mu) - nu*DM1b + gamma*DM2c).H
    
    psi = np.zeros((Nx,Nt), dtype=complex)
    
    psi[:,-1] = psiT
    
    for it in reversed(range(Nt-1)):
        psi[:,it] = CN_L_adj_advance(psi[:,it+1],LH,q[:,it],q[:,it+1],dt)
        
    return psi

def CN_NLf_advance(q,L,fa,fb,dt):

    I = sp.eye(L.shape[0])
    
    a = I - 0.5*dt*L
    A = I + 0.5*dt*L
    b = A @ q + 0.5*dt*(fa+fb) - dt*abs(q)**2*q

    return sp.linalg.spsolve(a,b)

def CN_NL_advance(q,L,dt):

    Z = 0.0*q

    return CN_NLf_advance(q,L,Z,Z,dt)

def CN_Lf_advance(q,L,fa,fb,dt):

    I = sp.eye(L.shape[0])

    # next step (a) is implicit
    a = I - 0.5*dt*L
    # current step (b) is explicit
    A = I + 0.5*dt*L
    b = A @ q + 0.5*dt*(fa+fb)

    return sp.linalg.spsolve(a,b)

def CN_L_advance(q,L,dt):

    Z = 0.0*q

    return CN_Lf_advance(q,L,Z,Z,dt)

def CN_NLf_adj_advance(psi,LH,qa,qb,dt):
    
    I = sp.eye(LH.shape[0])
    
    LH_a = LH + 2*sp.diags(abs(qa))**2
    a = I - 0.5*dt*LH_a
    
    LH_b = LH + 2*sp.diags(abs(qb))**2
    A = I + 0.5*dt*LH_b
    b = A @ psi + 0.5*dt*(qa+qb)

    return sp.linalg.spsolve(a,b)

def CN_Lf_adj_advance(psi,LH,qa,qb,dt):
    
    I = sp.eye(LH.shape[0])
    
    # next step (a) is implicit
    a = I - 0.5*dt*LH
     
    # current step (b) is explicit
    A = I + 0.5*dt*LH
    
    # forcing is always explicit
    b = A @ psi + 0.5*dt*(qa+qb)

def CN_L_adj_advance(psi,LH,dt):

    Z = 0.0*psi
    
    CN_Lf_adj_advance(psi,LH,Z,Z,dt)

    return sp.linalg.spsolve(a,b)

