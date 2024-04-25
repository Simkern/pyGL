#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:18:18 2024

@author: skern
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from time import time as tm
import sys

from scipy import linalg
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from git.solvers.lyapunov import LR_OSI
from git.core.utils import pmat, pvec

def rn(X,Xref):
    # Norm proposed in Mena et al
    n = X.shape[0]
    return np.linalg.norm(X - Xref)/n

def rn2(X,Xref):
    # Alternative relative norm (not useful in homogeneous case)
    return np.linalg.norm(X - Xref)/np.linalg.norm(Xref)

# direct Runge-Kutta time integration
def Xdot(t,Xv,A,Q):
    n = A.shape[0]
    X = Xv.reshape((n,n))
    dXdt = A @ X + X @ A.T + Q
    return dXdt.flatten()

plt.close("all")

## parameters
eps    = 1e-12
n      = 8            # number of points per dimension
rk_rhs = 5             # rank of the RHS inhomogeneity
rk_X0  = 10            # rank of the initial condition

I   = np.eye(n)
h   = 1/n**2
At  = np.diag(-2/h*np.ones((n,)),0) + \
      np.diag(np.ones((n-1,))/h,1) + \
      np.diag(np.ones((n-1,))/h,-1)
A   = np.kron(At,I) + np.kron(I,At)
N   = A.shape[0]


# Generate RHS
B = np.random.random_sample((N,rk_rhs))
Q = B @ B.T

# Generate initial condition
s0    = np.random.random_sample((rk_X0,))
U0, _ = linalg.qr(np.random.random_sample((N, rk_X0)),mode='economic')
S0    = np.diag(s0);
X0    = U0 @ S0 @ U0.T

'''
The reference will be the direct solution of the Lyapunov equation
'''
etime = tm()
# direct solve of A @ X + X @ A.T = -B @ B.T
Xref = linalg.solve_continuous_lyapunov(A, -Q)
print(f'Direct solve:    etime = {tm()-etime:5.2f}')

U,S,_ = linalg.svd(Xref)
fig = plt.figure(1)

'''
In order to find the integration time to steady state we integrate
with RK for different solver tolerances and compare to the reference
To see the evolution of the errror for each solver tolerance,
we integrate the same initial condition up to Tend = 20.0 in four steps
of Dt = 5.0 and compute the error after each part.
'''
Tend = 0.2
tspan = (0,Tend)
Nrep = 5
tolv = np.logspace(-6,-12,4)

Xa = np.empty((N,N,len(tolv),Nrep))

fig = plt.figure(2)
for it, tol in enumerate(tolv):
    X00 = X0
    erel = np.empty((Nrep+1,1))
    erel.fill(np.NaN)
    erel[0] = rn(X00,Xref)
    ts = 0.0
    te = Tend
    flag = False
    for i in range(Nrep):
        etime = tm()
        tolrk = tol
        sol = solve_ivp(Xdot,tspan,X00.flatten(),args=(A,Q), atol=tolrk, rtol=tol)
        X = sol.y[:,-1].reshape(A.shape)
        Xa[:,:,it,i] = X
        X00 = X
        erel[i] = rn(X,Xref)
        print(f'RK T in [{ts:4.2f},{te:4.2f}], tol={tolrk:.0e}:   etime = {tm()-etime:5.2f}   rel error: {rn(X,Xref):.4e}')
        ts += Tend
        te += Tend
        if ((erel[i] < tol and not flag) or (i == Nrep - 1)):
            flag = True
            #te -= Tend
            # we will integrate up to 'te' in the comparison with DLRA
            U,S,_ = linalg.svd(X)
            fig = plt.figure(1)
            plt.scatter(np.arange(N)+1,S,s=60,marker='o',
                        label=f'T = {te+Tend:4.1f}, tol = {tolrk:.1e}')
    print('')
    fig = plt.figure(2)
    plt.semilogy(np.arange(Nrep+1)*Tend+Tend,erel,label=f'tol={tolrk:.0e}',marker='o')
    plt.axhline(tol,color='k',linestyle='--',)
plt.title('Relative error ||X-Xref||/||Xref||')
plt.xlabel('Integration time [s]')
plt.ylabel('error vs. direct solve')
plt.legend()

fig = plt.figure(1)
plt.scatter(np.arange(N)+1,S,s=50,color='k',marker='*',label='clyap')
plt.yscale('log')
plt.xlim(1,90)
plt.ylim(1e-14,100)
plt.legend()

# compare RK45 & Stuart-Bartels to LR_OSI
etime = tm()
Tend = 0.1 # Nrep*Tend #te
tspan = (0,Tend)
tol = 1e-12
sol = solve_ivp(Xdot,tspan,X0.flatten(),args=(A,Q), atol=tol, rtol=tol)
Xrk = sol.y[:,-1].reshape(A.shape)
Urk,Srk,Vrkh = linalg.svd(Xrk)
print(f'RK solution time {tm()-etime:5.2f}   rel error: {rn(Xref,Xrk):.4e}')

rkv = [ 2, 6, 10, 14 ]
#rkv = [ 2, 8, 14 ]
#tauv = np.logspace(-2, -5, 4)
#tauv = np.logspace(-1, -5, 5)
tauv = np.logspace(-1, -5, 5)

sv = []
for i, rk in enumerate(rkv):
    erelrk = []
    erelsb = []
    for j, tau in enumerate(tauv):
        etime = tm()
        U,S,res = LR_OSI(A, B, X0, Tend, tau, 'rk', rk, verb=0)
        X = U @ S @ U.T
        sv.append(np.diag(S))
        erelrk.append(rn(X,Xrk))
        erelsb.append(rn(X,Xref))
        print(f'   dt={tau:.0e}:  etime = {tm()-etime:5.2f}   rel error: {rn(X,Xrk):.4e}')
    print('')
    fig = plt.figure(3)
    plt.loglog(tauv,erelrk,label=f'rk={rk:d}',marker='o')
    fig = plt.figure(4)
    plt.loglog(tauv,erelsb,label=f'rk={rk:d}',marker='o')

fig = plt.figure(3)
plt.title('Relative error ||X-Xrk||/||Xrk||')
plt.xlabel('dt')
plt.ylabel(f'error vs. RK45({tol:.0e})')
plt.legend()
fig = plt.figure(4)
plt.title('Relative error ||X-Xref||/||Xref||')
plt.xlabel('dt')
plt.ylabel('error vs. Stuart-Bartels')
plt.legend()


import numpy as np
import time
import sys

from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import lu_factor, lu_solve, solve, solve_lyapunov, svd, qr, inv, norm, expm

sys.path.append('..')

from git.solvers.lyap_utils import residual, M_ForwardMap, G_ForwardMap
from git.core.utils import pmat

def lrcfadi(A,B,p,stop_criterion,criterion_type,Xref):
    
    if criterion_type == 'niter':
        res_tol = False    
        niter = stop_criterion
        tol = None
    elif criterion_type == 'tol':
        res_tol = True
        niter = 1000
        tol = stop_criterion
    else:
        raise ValueError('Unsupported stop criterion')
    
    nord = None # defaults to 'fro' (also for vectors)
    l = p.size
    res_step = max(l,20)
    I = np.eye(A.shape[0])
    
    etime = time.time()
    lu_v = []
    for i in range(l):
        lu, piv = lu_factor(A + p[i]*I)
        lu_v.append((lu,piv))
    etime_LU = time.time() - etime

    Q,R  = qr(B,mode='economic')
            
    nrm0 = norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    is_breakdown = False
    
    etime_res   = []
    etime_CFADI = 0
    
    for i in range(niter):
        etime = time.time()
        if i == 0:
            V = np.sqrt(-2*p[ip])*lu_solve(lu_v[ip],B)
            Z = V
        else:
            pold = p[ip]        # keep old shift
            ip = (ip + 1) % l   # update shift index
        
            f1 = np.sqrt(-p[ip])/np.sqrt(-pold)
            f2 = np.real(p[ip] + pold)
             
            V = f1*(V - f2*lu_solve(lu_v[ip],V))
            
            Z = np.column_stack([ Z , V ])
        etime_CFADI = etime_CFADI + time.time() - etime
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            ncols.append(V.shape[1])
            Q, R, nrm = residual(V, A, Q, R, ncols, nord)
            res.append(nrm)
            res_rel.append(nrm/nrm0)
            ires.append(i) 
            nrmx.append(norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(norm(V, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
            if nrm == 0:
                is_breakdown = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime LU :    {etime_LU:10.6f}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}\n')
    else:
        if is_breakdown:
            print(f'  QR GS breakdown after {i} iterations.')
        else:
            print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime LU :    {etime_LU:10.6f}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}\n')
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel

def lrcfadic_r(A,B,pin,stop_criterion,criterion_type,Xref):
    
    if criterion_type == 'niter':
        res_tol = False    
        niter = stop_criterion
        tol = None
    elif criterion_type == 'tol':
        res_tol = True
        niter = 200
        tol = stop_criterion
    else:
        raise ValueError('Unsupported stop criterion')
    
    nord = None # defaults to 'fro' (also for vectors)
    I = np.eye(A.shape[0])
    l = pin.size
    
    etime = time.time()
    is_real = np.isreal(pin)   
    lu_v = []
    p_v  = np.array([])
    for i, (p, is_r) in enumerate(zip(pin,is_real)):
        if is_r:
            Atmp = A + np.real(p)*I
        else:
            s_i = 2*np.real(p)
            t_i = np.abs(p)**2
            Atmp = A @ A + s_i*A + t_i*I
        lu, piv = lu_factor(Atmp)
        lu_v.append((lu,piv))
        p_v = np.append(p_v,p)
        
    etime_LU = time.time() - etime
      
    res_step = max(l,20)   
    
    Q,R  = qr(B,mode='economic')
            
    nrm0 = norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    is_breakdown = False
    
    etime_res   = []
    etime_CFADI = 0
    
    # initialisation
    p1         = p_v[0]
    p1_is_real = np.isreal(p1)
    if p1_is_real:
        p1 = np.real(p1)
        
    if p1_is_real:
        q   = np.sqrt(-2*p1)
        V1t = lu_solve(lu_v[0], B)
        V2t = 0*V1t #dummy
        Z   = q*V1t
    else: # p1 complex
        q1  = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2  = 2*np.sqrt(-np.real(p1))
        V1t = lu_solve(lu_v[0], B)
        V1  = q1*V1t
        V2t = A @ V1t
        V2  = q2*V2t
        Z   = np.column_stack([ V1, V2 ])
    
    p_old   = p1
    V1t_old = V1t
    V2t_old = V2t
    
    # iteration
    ip = 0
    for i in range(1,niter):
        etime = time.time()
        
        ip = (ip + 1) % l
        
        p             = p_v[ip]
        p_is_real     = np.isreal(p)
        if p_is_real:
            p = np.real(p)
        p_old_is_real = np.isreal(p_old)
                  
        if p_is_real:
            q  = np.sqrt(-2*p)
            if p_old_is_real:
                f1  = (p + p_old)
                V1t = V1t_old - f1*lu_solve(lu_v[ip], V1t_old)
            else: # p_old complex
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                V1t = V2t_old - f1*V1t_old + f2*lu_solve(lu_v[ip], V1t_old)
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                Vtmp = A @ V1t_old - p_old*V1t_old
                V1t  = lu_solve(lu_v[ip], Vtmp)
            else: # p_old complex
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp = f1*V1t_old - f2*V2t_old
                V1t  = V1t_old + lu_solve(lu_v[ip], Vtmp)
            V2t  = A @ V1t
            V1   = q1*V1t
            V2   = q2*V2t
            Vnew = np.column_stack([ V1, V2 ]) 
        Z = np.column_stack([ Z, Vnew ])
            
        p_old   = p
        V1t_old = V1t
        V2t_old = V2t
                    
        etime_CFADI = etime_CFADI + time.time() - etime
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            try:
                newcols = Vnew.shape[1]
            except IndexError:
                newcols = 1
            ncols.append(newcols)
            Q, R, nrm = residual(Vnew, A, Q, R, ncols, nord)
            res.append(nrm)
            res_rel.append(nrm/nrm0)
            ires.append(i)
            nrmx.append(norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(norm(Vnew, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
            if nrm == 0:
                is_breakdown = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime LU :    {etime_LU:10.6f}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}\n')
    else:
        if is_breakdown:
            print(f'  QR GS breakdown after {i} iterations.')
        else:
            print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime LU :    {etime_LU:10.6f}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}\n')
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel

def lrcfadic_r_gmres(A,B,pin,stop_criterion,criterion_type,Xref,stol):
    
    nord = None # defaults to 'fro' (also for vectors)
    I = np.eye(A.shape[0])
    l = pin.size
    
    def gmres_solve(A,b):
        etime = time.time()
        x, info = gmres(A, b, tol=stol, maxiter=2000)
        etime = time.time() - etime
        return x, info, etime
    
    if criterion_type == 'niter':
        res_tol = False    
        niter   = stop_criterion
        rounds  = np.ceil(niter/l)
        nmax    = rounds*l
        tol     = None
    elif criterion_type == 'tol':
        res_tol = True
        rounds  = 20
        nmax    = rounds*l
        niter   = nmax
        tol     = stop_criterion
    else:
        raise ValueError('Unsupported stop criterion')
    
    is_real = np.isreal(pin)   
    A_v = []
    p_v = np.array([])
    singlestep = []
    for i, (p, is_r) in enumerate(zip(pin,is_real)):
        if is_r:
            Atmp = A + np.real(p)*I
            singlestep.append(True)
        else:
            s_i = 2*np.real(p)
            t_i = np.abs(p)**2
            Atmp = A @ A + s_i*A + t_i*I
            singlestep.append(False)
        A_v.append(Atmp)
        p_v = np.append(p_v,p)
      
    res_step = max(l,20)   
    
    Q,R  = qr(B,mode='economic')
            
    nrm0 = norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    is_breakdown = False
    
    etime_res   = []
    etime_CFADI = 0
    etime_gmres = np.zeros((nmax,))
    
    # initialisation
    p1         = p_v[0]
    p1_is_real = np.isreal(p1)
    if p1_is_real:
        p1 = np.real(p1)
        
    if p1_is_real:
        q   = np.sqrt(-2*p1)
        V1t, info, etime_gmres[0] = gmres_solve(A_v[0], B)
        V2t = 0*V1t #dummy
        Z   = q*V1t
    else: # p1 complex
        q1  = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2  = 2*np.sqrt(-np.real(p1))
        V1t, info, etime_gmres[0] = gmres_solve(A_v[0], B)
        V1  = q1*V1t
        V2t = A @ V1t
        V2  = q2*V2t
        Z   = np.column_stack([ V1, V2 ])
    
    p_old   = p1
    V1t_old = V1t
    V2t_old = V2t
    
    # iteration
    ip = 0
    for i in range(1,niter):
        etime = time.time()
        
        ip = (ip + 1) % l
        
        p             = p_v[ip]
        p_is_real     = np.isreal(p)
        if p_is_real:
            p = np.real(p)
        p_old_is_real = np.isreal(p_old)
                  
        if p_is_real:
            q  = np.sqrt(-2*p)
            if p_old_is_real:
                f1  = (p + p_old)
                Vtmp, info, etime_gmres[i] = gmres_solve(A_v[ip], V1t_old)
                V1t = V1t_old - f1*Vtmp
            else: # p_old complex
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                Vtmp, info, etime_gmres[i] = gmres_solve(A_v[ip], V1t_old)
                V1t = V2t_old - f1*V1t_old + f2*Vtmp
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                Vtmp = A @ V1t_old - p_old*V1t_old
                V1t, info, etime_gmres[i] = gmres_solve(A_v[ip], Vtmp)           
            else: # p_old complex
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp = f1*V1t_old - f2*V2t_old
                Vtmp2, info, etime_gmres[i] = gmres_solve(A_v[ip], Vtmp)
                V1t  = V1t_old + Vtmp2
            V2t  = A @ V1t
            V1   = q1*V1t
            V2   = q2*V2t
            Vnew = np.column_stack([ V1, V2 ]) 
        Z = np.column_stack([ Z, Vnew ])
            
        p_old   = p
        V1t_old = V1t
        V2t_old = V2t
                    
        etime_CFADI = etime_CFADI + time.time() - etime
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            try:
                newcols = Vnew.shape[1]
            except IndexError:
                newcols = 1
            ncols.append(newcols)
            Q, R, nrm = residual(Vnew, A, Q, R, ncols, nord)
            res.append(nrm)
            res_rel.append(nrm/nrm0)
            ires.append(i)
            nrmx.append(norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(norm(Vnew, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
            if nrm == 0:
                is_breakdown = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}')
        print(f'  etime gmres : {np.sum(etime_gmres):10.6f}   ',end='')
        print('  min/avg/max :',end='')
        print(f' {    min(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {np.mean(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {    max(etime_gmres[etime_gmres > 0]):10.6f}\n')
    else:
        if is_breakdown:
            print(f'  QR GS breakdown after {i} iterations.')
        else:
            print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}')
        print(f'  etime gmres : {np.sum(etime_gmres):10.6f}   ',end='')
        print('  min/avg/max :',end='')
        print(f' {    min(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {np.mean(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {    max(etime_gmres[etime_gmres > 0]):10.6f}\n')    
    
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel

def lrcfadic_r_gmres_matvec(A,B,M,pin,stop_criterion,criterion_type,Xref,stol):
    
    M = aslinearoperator(M)
    
    def gmres_solve(A,b):
        etime = time.time()
        x, info = gmres(A, b, M=M, tol=stol, maxiter=2000)
        etime = time.time() - etime
        return x, info, etime
    
    nord = None# defaults to 'fro' (also for vectors)
    I = np.eye(A.shape[0])
    l = pin.size
    
    if criterion_type == 'niter':
        res_tol = False    
        niter   = stop_criterion
        rounds  = np.ceil(niter/l)
        nmax    = rounds*l
        tol     = None
    elif criterion_type == 'tol':
        res_tol = True
        rounds  = 20
        nmax    = rounds*l
        niter   = nmax
        tol     = stop_criterion
    else:
        raise ValueError('Unsupported stop criterion')
    
    is_real = np.isreal(pin)   
    A_v = []
    p_v = np.array([])
    singlestep = []
    for i, (p, is_r) in enumerate(zip(pin,is_real)):
        if is_r:
            Atmp = A + np.real(p)*I
            singlestep.append(True)
        else:
            s_i = 2*np.real(p)
            t_i = np.abs(p)**2
            Atmp = A @ A + s_i*A + t_i*I
            singlestep.append(False)
        A_v.append(Atmp)
        p_v = np.append(p_v,p)
      
    res_step = max(l,20)   
    
    Q,R  = qr(B,mode='economic')
            
    nrm0 = norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    is_breakdown = False
    
    etime_res   = []
    etime_CFADI = 0
    etime_gmres = np.zeros((nmax,))
    
    # initialisation
    p1         = p_v[0]
    p1_is_real = np.isreal(p1)
    if p1_is_real:
        p1 = np.real(p1)
        
    if p1_is_real:
        q   = np.sqrt(-2*p1)   
        V1t, info, etime_gmres[0] = gmres_solve(A_v[0], B)
        V2t = 0*V1t #dummy
        Z   = q*V1t
    else: # p1 complex
        q1  = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2  = 2*np.sqrt(-np.real(p1))
        V1t, info, etime_gmres[0] = gmres_solve(A_v[0], B)
        V1  = q1*V1t
        V2t = A @ V1t
        V2  = q2*V2t
        Z   = np.column_stack([ V1, V2 ])
    if not info == 0:
        print('Step 1: GMRES unconverged.')
    
    p_old   = p1
    V1t_old = V1t
    V2t_old = V2t
    
    # iteration
    ip = 0
    for i in range(1,niter):
        etime = time.time()
        
        ip = (ip + 1) % l
        
        p             = p_v[ip]
        p_is_real     = np.isreal(p)
        if p_is_real:
            p = np.real(p)
        p_old_is_real = np.isreal(p_old)
                  
        if p_is_real:
            q  = np.sqrt(-2*p)
            if p_old_is_real:
                f1  = (p + p_old)
                Vtmp, info, etime_gmres[i] = gmres_solve(A_v[ip], V1t_old)
                V1t = V1t_old - f1*Vtmp
            else: # p_old complex
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                Vtmp, info, etime_gmres[i] = gmres_solve(A_v[ip], V1t_old)
                V1t = V2t_old - f1*V1t_old + f2*Vtmp
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                Vtmp = A @ V1t_old - p_old*V1t_old
                V1t, info, etime_gmres[i] = gmres_solve(A_v[ip], Vtmp)     
            else: # p_old complex
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp = f1*V1t_old - f2*V2t_old
                Vtmp2, info, etime_gmres[i] = gmres_solve(A_v[ip], Vtmp)
                V1t  = V1t_old + Vtmp2
            V2t  = A @ V1t
            V1   = q1*V1t
            V2   = q2*V2t
            Vnew = np.column_stack([ V1, V2 ]) 
        Z = np.column_stack([ Z, Vnew ])
        if not info == 0:
            print(f'Step {i+1:3d}: GMRES unconverged.')
            
        p_old   = p
        V1t_old = V1t
        V2t_old = V2t
                    
        etime_CFADI = etime_CFADI + time.time() - etime
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            try:
                newcols = Vnew.shape[1]
            except IndexError:
                newcols = 1
            ncols.append(newcols)
            Q, R, nrm = residual(Vnew, A, Q, R, ncols, nord)
            res.append(nrm)
            res_rel.append(nrm/nrm0)
            ires.append(i)
            nrmx.append(norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(norm(Vnew, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
            if nrm == 0:
                is_breakdown = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}')
        print(f'  etime gmres : {np.sum(etime_gmres):10.6f}   ',end='')
        print('  min/avg/max :',end='')
        print(f' {    min(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {np.mean(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {    max(etime_gmres[etime_gmres > 0]):10.6f}\n')
    else:
        if is_breakdown:
            print(f'  QR GS breakdown after {i} iterations.')
        else:
            print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_F/||X_ref||_F = {nrmx[-1]}')
        print(f'  etime solve : {etime_CFADI:10.6f}')
        print(f'  etime res :   {np.sum(etime_res):10.6f}   ',end='')
        print(f'  min/avg/max : {min(etime_res):10.6f} {np.mean(etime_res):10.6f} {max(etime_res):10.6f}')
        print(f'  etime gmres : {np.sum(etime_gmres):10.6f}   ',end='')
        print('  min/avg/max :',end='')
        print(f' {    min(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {np.mean(etime_gmres[etime_gmres > 0]):10.6f}',end='')
        print(f' {    max(etime_gmres[etime_gmres > 0]):10.6f}\n')    
    
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel

def kpik(A,B,k_max,tol,tolY):
    
    # Based on kpik.m avalible from V. Simoncini's website (http://www.dm.unibo.it/~simoncin/software.html)
    # Essentially a translation of the matlab code to python

    # Approximately solve
    #       A X + X A' + BB' = 0
    # by means of the extended Krylov subspace method
    # ARGUMENTS
    #     'A'     : coeff matrix, A < 0
    #     'B'     : factor of rhs,   nxm matrix with m << n
    #     'k_max' : max Krylov subspace dimension
    #     'tol'   : stopping tolerance based on the backwards error,
    #               with stopping criterion
    #                || A X + X A'- BB'||
    #               ----------------------   < tol
    #               ||BB'|| + ||A|| ||X||
    #              computed in a cheap manner.
    #     'tolY'  : threshold for a posteriori rank reduction of the 
    #               resulting Y matrix    
    # Output
    #     'Z'     : solution factor s.t. X = Z Z'
    #     'err2'  : history of scaled residual, as above
    #     'etime' : elapsed time
    
    # initialisation
    etime = time.time()
    # sizes & constants
    n, sh = B.shape
    s     = 2*sh
    sqrt2 = np.sqrt(2)
    rho   = 1 #dummy
    # norms
    normB = np.linalg.norm(B, 'fro')**2
    normA = np.linalg.norm(A, 'fro')
    # variables
    H    = np.zeros(((k_max+1)*s,k_max*s))
    T    = np.zeros(((k_max+1)*s,k_max*s))
    L    = np.zeros(((k_max+1)*s,k_max*s))
    err2 = np.zeros(k_max)
    odds = []

    # compute LU factorisation of A
    luA,piv = lu_factor(A)

    # initialise Krylov basis
    V1 = np.block([ B , lu_solve((luA,piv),B ) ])
    # Orthogonalize
    U,R  = qr(V1,mode='economic')
    Rinv = inv(R)
    R    = R[:sh,:sh]
    R2   = R @ R.T 

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
        Up[:, sh:] = lu_solve((luA,piv), U[:, jsh:js])
        
        # orthogonalise new vector wrt. current Krylov basis, add to Hessenberg matrix
        for l in range(2):      # MGS
            k_min = max(0, j - k_max - 1)
            for kk in range(k_min, j + 1):
                k1 = kk * s
                k2 = (kk+1) * s
                proj              = U[:, k1:k2].T @ Up
                H[k1:k2, jms:js] += proj
                Up               -= U[:, k1:k2] @ proj

        # orthogonalise new vectors wrt. to each other, add subdiagonal block to Hessenberg matrix
        if j < k_max:
            Up, H[js1:j1s, jms:js] = qr(Up, mode='economic')
            Hinv                   = inv(H[js1:j1s, jms:js])

        # determine the coefficient matrix for the projected problem
        # NOTE: avoids the explicit multiplication with A
        I = np.eye(js + s)
        if j == 0:
            # A/B = (B.T\A.T).T
            V1 = solve(Rinv[:sh, :sh].T,   H[:s+sh, :sh].T).T
            V2 = solve(Rinv[:sh, :sh].T,np.eye(s+sh, sh).T).T
            L[:s+sh,:sh] = np.block([ V1, V2 ]) @ Rinv[:s, sh:s]
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
                                   ) @ Hinv[sh:s, sh:s]

        rho = inv(Hinv[:sh, :sh]) @ Hinv[:sh, sh:s]

        # Solve the projected problem using Stuart-Bartels
        Y = solve_lyapunov(T[:js, :js], np.eye(js, sh) @ R2 @ np.eye(sh,js))
        # Ensure that the result is symmetric
        Y = (Y + Y.T) / 2

        # Compute residual
        cc = np.block([H[js1:j1s, js-s:js-sh], L[js1:j1s, j*sh:(j+1)*sh]])
            
        normX = norm(Y, 'fro')
        err2[j] = sqrt2 * norm(cc @ Y[js-s:js, :], 'fro') / (normB + normA * normX)

        if err2[j] < tol:
            break
        else:
            U = np.block([U,Up])

    # reduce rank of Y if possible
    uY, sY, _ = svd(Y)
    is_ = np.sum(np.abs(sY) > tolY)
    Y0 = uY[:, :is_] @ np.diag(np.sqrt(sY[:is_]))

    Z = U[:, :js] @ Y0
    k_eff = js
    err2 = err2[:j + 1]
    etime = time.time() - etime
    
    return Z, err2, k_eff, etime

def kpik_gmres(A,B,M,k_max,tol,tolY,stol):
    
    # Based on kpik.m avalible from V. Simoncini's website (http://www.dm.unibo.it/~simoncin/software.html)
    # Essentially a translation of the matlab code to python
    
    nmatvec = 0
    
    def gmres_solve(A,B):
        etime = time.time()
        counter = gmres_counter()
        x, info = gmres(A, B, M=M, tol=stol, maxiter=6000, callback=counter)
        etime = time.time() - etime
        return x, info, etime, counter
    
    class gmres_counter(object):
        def __init__(self):
            self.niter = 0
            self.rkv   = []
        def __call__(self, rk=None):
            self.niter += 1
            self.rkv.append(rk)
    
    # initialisation
    etime = time.time()
    # sizes & constants
    n, sh = B.shape
    s     = 2*sh
    sqrt2 = np.sqrt(2)
    rho   = 1 #dummy
    # norms
    normB = np.linalg.norm(B, 'fro')**2
    normA = np.linalg.norm(A, 'fro')
    # variables
    H    = np.zeros(((k_max+1)*s,k_max*s))
    T    = np.zeros(((k_max+1)*s,k_max*s))
    L    = np.zeros(((k_max+1)*s,k_max*s))
    err2 = np.zeros(k_max)
    odds = []

    # initialise Krylov basis
    rhs1, info, etime, counter = gmres_solve(A,B)
    if info != 0:
        print(counter.rkv[-1])
    rhs1 = rhs1.reshape(-1, 1)
    nmatvec += counter.niter   
    V1                = np.block([ B , rhs1 ])
    # Orthogonalize
    U,R  = qr(V1,mode='economic')
    Rinv = inv(R)
    R    = R[:sh,:sh]
    R2   = R @ R.T 

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
        AinvU, info, etime, counter = gmres_solve(A, U[:, jsh:js])
        if info != 0:
            print(counter.rkv[-1])
        Up[:, sh:] = AinvU.reshape(-1,1)
        nmatvec += counter.niter
        
        # orthogonalise new vector wrt. current Krylov basis, add to Hessenberg matrix
        for l in range(2):      # MGS
            k_min = max(0, j - k_max - 1)
            for kk in range(k_min, j + 1):
                k1 = kk * s
                k2 = (kk+1) * s
                proj              = U[:, k1:k2].T @ Up
                H[k1:k2, jms:js] += proj
                Up               -= U[:, k1:k2] @ proj

        # orthogonalise new vectors wrt. to each other, add subdiagonal block to Hessenberg matrix
        if j < k_max:
            Up, H[js1:j1s, jms:js] = qr(Up, mode='economic')
            Hinv                   = inv(H[js1:j1s, jms:js])

        # determine the coefficient matrix for the projected problem
        # NOTE: avoids the explicit multiplication with A
        I = np.eye(js + s)
        if j == 0:
            # A/B = (B.T\A.T).T
            V1 = solve(Rinv[:sh, :sh].T,   H[:s+sh, :sh].T).T
            V2 = solve(Rinv[:sh, :sh].T,np.eye(s+sh, sh).T).T
            L[:s+sh,:sh] = np.block([ V1, V2 ]) @ Rinv[:s, sh:s]
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
                                   ) @ Hinv[sh:s, sh:s]

        rho = inv(Hinv[:sh, :sh]) @ Hinv[:sh, sh:s]

        # Solve the projected problem using Stuart-Bartels
        Y = solve_lyapunov(T[:js, :js], np.eye(js, sh) @ R2 @ np.eye(sh,js))
        # Ensure that the result is symmetric
        Y = (Y + Y.T) / 2

        # Compute residual
        cc = np.block([H[js1:j1s, js-s:js-sh], L[js1:j1s, j*sh:(j+1)*sh]])
            
        normX = norm(Y, 'fro')
        err2[j] = sqrt2 * norm(cc @ Y[js-s:js, :], 'fro') / (normB + normA * normX)
        print(err2[j])

        if err2[j] < tol:
            break
        else:
            U = np.block([U,Up])

    # reduce rank of Y if possible
    uY, sY, _ = svd(Y)
    is_ = np.sum(np.abs(sY) > tolY)
    Y0 = uY[:, :is_] @ np.diag(np.sqrt(sY[:is_]))

    Z = U[:, :js] @ Y0
    k_eff = js
    err2 = err2[:j + 1]
    etime = time.time() - etime
    
    return Z, err2, k_eff, etime, nmatvec

def LR_OSI(A,B,X0,Tend,dtaim,rk_type,rk,torder=1,verb=1):
    
    if verb > 0:
        print('\nLow-rank operator-splitting method for Differential Lyapunov equations.\n')
    #else:
    #    print(f'LR_OSI {rk_type} = {rk}.')
    
    eps = 1e-12
    n = X0.shape[0]
    # generate inhomogeneity
    Q = B @ B.T
    nQ = np.linalg.norm(Q)
    # check rank of inhomogeneity
    Uq,Sq,_ = svd(Q)
    rkq = sum(Sq > eps)
    if verb > 0:
        print(f'Numerical rank of inhomogeneity B  ({n:d}x{n:d}): {rkq:3d} (tol={eps:.2e})')
    
    # check rank of initial data
    U,S,_ = svd(X0, full_matrices=False)
    rk0 = sum(S > eps)
    if verb > 0:
        print(f'Numerical rank of initial data  X0 ({n:d}x{n:d}): {rk0:3d} (tol={eps:.2e})')
    
    if isinstance(rk,str) and rk_type == 'sigma_tol':
        if rk < eps:
            stol = eps
        else:
            stol = rk
        S0   = np.diag(S[S > stol])
        rk   = len(S)
        if verb > 0:
            print(f'\nMode sigma_tol:    stol = {stol:.2e}')
            print(f'  Required rank of the initial data: {rk:d}\n')   
    elif int(rk)==rk and rk > 0:
        stol = eps
        if verb > 0:
            print('\nMode rank:')
            print(f'  Chosen rank: {rk:d}\n')
        S0 = np.diag(S[:rk])
    else:
        raise TypeError
    # pick orthonormal columns
    U0   = np.zeros((n,rk))
    #U0[:,:min(rk,rk0)]   = U[:,:min(rk,rk0)]
    #U0[:rk0,:rk0] = np.eye(rk0)
       
    res = []
    
    nt    = int(np.ceil(Tend/dtaim))
    dt    = Tend/nt
    tspan = np.linspace(0, Tend, num=nt, endpoint=True)
    
    # precompute matrix exponential
    if torder == 1:
        exptA = expm(dt*A)
    elif torder == 2:
        exptA = expm(0.5*dt*A)
    else:
        print('torder >= 2 currently not implemented.')
        raise ValueError
        return
    
    if verb > 0:
        print(f'Begin iteration:   0 --> {Tend:4.2f}      dt = {dt:.5e}')
    etime = time.time()
    iprint = int(np.floor(nt/10))
    for it in range(nt):
        if not iprint ==0 and (it % iprint == 0 or it == nt-1):
            if verb > 0:
                print(f'  Step {it+1:4d}: t = {tspan[it]:4.2f}')
            X = U0 @ S0 @ U0.T
            res.append(np.linalg.norm(A @ X + X @ A.T + Q)/nQ)
        if torder == 1:
            #pmat(U0, 'U0 pre')
            #print(f'dt = {dt}')
            U1A, S1A = M_ForwardMap(A, U0, S0, dt, None)
            #pmat(U0, 'U0 post M')
            U0, S0   = G_ForwardMap(U1A, S1A, Q, dt)
            #pmat(U0, 'U0 post G')
        elif torder == 2:
            U1A, S1A = M_ForwardMap(A, U0, S0, dt/2, exptA)
            U2, S2   = G_ForwardMap(U1A, S1A, Q, dt)
            U0, S0   = M_ForwardMap(A, U2, S2, dt/2, exptA)
            
        #if it == 0:
        #    sys.exit()
         
    etime = time.time() - etime
    if verb > 0:
        print(f'Elapsed time:   {etime:4.2f} seconds.')
    Uout = U0
    Sout = S0
    return Uout, Sout, res
    