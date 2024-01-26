import numpy as np
import math as m
import sys

from scipy import linalg
from numpy import mean as am
from scipy.stats import gmean as gm
from scipy.linalg import expm, qr

sys.path.append('../core')

from git.solvers.arnoldi import arn

from utils import en

def check_shifts(p_v):
    
    status = 0
    n = p_v.size
    
    is_real = np.imag(p_v) < 1e-12
    
    for i, is_r in enumerate(is_real):
        if is_r:
            p_v[i] = np.real(p_v[i])
        
    if n == 1:
       if not is_real[0]:
            status = -1
    else:
        if not all(is_real):
            status = 1
            is_first = True
            for (i,(curr_is_real,next_is_real)) in enumerate(zip(is_real[:-1],is_real[1:])):
                if not curr_is_real and is_first:
                    is_first = False
                    if not p_v[i] == p_v[i+1].conj():
                        status = -1
                elif not curr_is_real and not is_first:
                        is_first = True
            if status >= 0 and not is_real[-1] and not p_v[-1] == p_v[-2].conj():
               status = -1
            
    if status < 0:
        print('Shifts must appear in complex conjugate pairs!\n')

    return p_v, status, is_real
               
def updateQR(W,Q,R):
    eps = 1e-20
    # Update and extend QR factorisation by adding the columns of W
    ncolQ = Q.shape[1]
    ncolW = W.shape[1]
    for iw in range(ncolW):
        proj = np.zeros((ncolQ,1))   # needs to be a column vector for cat
        wcol = W[:,iw]
        # MGS
        for iq in range(ncolQ):      # Q grows at every outer iteration 
            qcol     = Q[:,iq]
            proj[iq] = np.dot(qcol , wcol)
            wcol    -= proj[iq] * qcol
        beta = en(wcol)
        if beta < 1e-8:              # 2nd pass fixes orthonormality in edge cases
            proj2 = np.zeros((ncolQ,1))
            for iq in range(ncolQ):
                qcol      = Q[:,iq]
                proj2[iq] = np.dot(qcol , wcol)
                wcol     -= proj2[iq] * qcol
            proj += proj2
            beta  = en(wcol)
        # add column to basis and inflate R
        if beta < eps:
            print('Breakdown: Newest iterate is linearly dependent!')
            status = -1
            break
        elif m.isinf(beta):
            print('Breakdown: beta = Inf')
            status = -2
            break
        elif m.isnan(beta):
            print('Breakdown: beta = NaN')
            status = -3
            break
        else:
            Q = np.column_stack([ Q, wcol/beta ])
            R = np.block([[ R , proj ] , [ np.zeros((1,ncolQ)) , beta ]])
            ncolQ = ncolQ + 1
            status = 0
            
    return Q, R, status

def residual(V, A, Q, R, ncols, nord):
    
    W = np.column_stack([ A.T @ V , V ])  
    
    Q, R, status = updateQR(W, Q, R)
    
    if status == 0:
        # compute | R P R^T |_fro where P amounts to switching column blocks
        j2 = ncols[0]    # m = B.shape[1]
        RP  = np.copy(R)
        for nc in ncols[1:]: # width of the column range to swap in iterate
            # identify contiguous column ranges of interest
            i1 = j2; j1 = j2 + nc; c1 = slice(i1,j1)
            i2 = j1; j2 = j1 + nc; c2 = slice(i2,j2)
            # swap
            tmp = RP[:,c1].copy()
            RP[:,c1] = RP[:,c2]
            RP[:,c2] = tmp
        nrm = linalg.norm(RP @ R.T, ord=nord)
    else:
        nrm = 0
        
    return Q,R,nrm

def get_opt_shifts(a0,b0,c0,d0,n):
    
    th0 = (a0*b0 - c0*d0)/(a0 + b0 + c0 + d0)
    
    ab0p = (a0 - th0 , b0 - th0)
    cd0p = (c0 + th0 , d0 + th0)
    
    ab  = [ ( gm(ab0p) , am(ab0p) ) ]
    cd  = [ ( gm(cd0p) , am(cd0p) ) ]
    
    """
    print(f'[a 0,b 0] = [{a0:9.5f},{b0:9.5f}], ',end='')
    print(f'[c 0,d 0] = [{c0:9.5f},{d0:9.5f}], th0 = {th0}')
    print(f"[a'0,b'0] = [{ab0p[0]:9.5f},{ab0p[1]:9.5f}], ",end='')
    print(f"[c'0,d'0] = [{cd0p[0]:9.5f},{cd0p[1]:9.5f}]")
    """
    
    if n == 0:
        abp = ab0p
    else:
        th  = []; abp = []; cdp = []
        for i in range(n):
            
            th.append( ab[i][0]*( ab[i][1] - cd[i][1] )/( 2*ab[i][0] + ab[i][1] + cd[i][1] ) )
            abp.append( (ab[i][0] - th[i] , ab[i][1] - th[i]) )
            cdp.append( (cd[i][0] + th[i] , cd[i][1] + th[i]) )
            
            if not i == n-1:        
                ab.append( ( gm(abp[i]) , am(abp[i]) ) )
                cd.append( ( gm(cdp[i]) , am(cdp[i]) ) )         
            """
            print(f'[a {i+1},b {i+1}] = [{ab[i][0]:9.4f},{ab[i][1]:9.4f}], ',end='')
            print(f'[c {i+1},d {i+1}] = [{cd[i][0]:9.4f},{cd[i][1]:9.4f}], ',end='')
            print(f' th{i+1} = {th[i]}')
            print(f"[a'{i+1},b'{i+1}] = [{abp[i][0]:9.4f},{abp[i][1]:9.4f}], ",end='')
            print(f"[c'{i+1},d'{i+1}] = [{cdp[i][0]:9.4f},{cdp[i][1]:9.4f}]")
            """
    
    opL = [ gm(abp[-1]) ]
    opG = [ gm(abp[-1]) ]
    for i in reversed(range(n)):    
        oL = []; oG = []  
        for (lp,gp) in zip(opL,opG):
            oL.append( lp - th[i] )
            oG.append( gp + th[i] )
            
        an2 = ab[i][0]**2
        opL = []; opG = []
        for (l,g) in zip(oL,oG):
            tmp = l + np.sqrt( l**2 - an2)
            opL.extend( [ tmp, an2/tmp ] )
            tmp = g + np.sqrt( g**2 - an2)
            opG.extend( [ tmp, an2/tmp ] )
    # Generate final shifts
    oL = []; oG = []
    for (lp,gp) in zip(opL,opG):
        oL.append( lp - th0 )
        oG.append( gp + th0 )
    
    #print('Final optimal shifts:')
    if [a0,b0] == [c0,d0]:
        #print(' omega = ',end='')
        #print(", ".join(format(shift, "7.2f") for shift in sorted(oL, reverse=True)))
        oG = []
    #else:
        #print(' omega_V = ',end='')
        #print(", ".join(format(shift, "7.2f") for shift in sorted(oL, reverse=True)))
        #print(' omega_H = ',end='')
        #print(", ".join(format(shift, "7.2f") for shift in sorted(oG, reverse=True)))
    
    return np.array(oL),np.array(oG)

def M_ForwardMap(A,U,S,tau,exptA=None,nkryl=None):
    """
    Direct solution of the (stiff) linear part of the Lyapunov equation
    in terms of the the low-rank factors U(t), S(t) over the time interval 
    tau
    
    M(t) = U(t) @ S(t) @ U.T(t) of rank r
    
    Mdot = A @ M(t) + M(t) @ A.T
    
    M(t+tau) = expm(tau*A) @ U(t) @ S(t) @ U.T(t) @ expm(tau*A.T)

    Parameters
    ----------
    A : np.array
        System matrix.
    U : np.array
        Low-rank basis of initial data U(t).
    S : np.array
        Low-rank approximation of initial data S(t).
    tau : float
        Incremental integration time horizon.
    exptA : np.array, optional
        If present, the array is considered to be the precomputed exponential 
        propagator. Otherwise, the exponential propagator is computed 
        explicitly on the fly
    nkryl : int, optional
        If present, it is considerd to be the number of arnoldi steps for the 
        Krylov approximation of the action of the exponential propagator
        NB: This option is ignored if exptA is passed
    Returns
    -------
    UA : np.array
        Low-rank basis of solution U(t + tau).
    SA : np.array
        Solution S(t + tau).
    """
    
    if exptA is None:
        if nkryl is None:
            U1 = expm(tau*A) @ U
        else:
            U1 = kryl_expm(A,U,nkryl,tau)
    else:
        U1     = exptA @ U
    UA, R = qr(U1,mode='economic')
    SA    = R @ S @ R.T
    
    return UA, SA

def G_ForwardMap(UA, SA, Q, tau):
    """
    Rank-preserving integration of the linear inhomogeneity Q that maintains 
    orthonormality of the basis U(t) over the time interval tau
    
    Y(t) = U(t) @ S(t) @ U.T(t) of rank r
    
    Ydot = Q @ V @ V.T - U @ U.T @ Q @ V @ V.T + U @ U.T @ Q

    Parameters
    ----------
    UA : np.array
        Low-rank basis of the solution of the stiff part of the Lyapunov
        equation.
    SA : np.array
        Low-rank solution of stiff part of the Lyapunov equation.
    Q : np.array
        Inhomogeneity in the Lyapunov equation.
    tau : float
        Incremental integration time horizon.

    Returns
    -------
    UA : np.array
        Low-rank basis of composed solution of the Lyapunov equation.
    SA : np.array
        Composed solution S(t + tau) of the Lyapunov equation
    """
    
    # solve Kdot = Q @ U1A with K0 = UA @ SA for one step tau
    K1 = UA @ SA + tau*(Q @ UA)
    # orthonormalise K1
    U1, Sh = qr(K1,mode='economic')
    # solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
    St = Sh - tau*( U1.T @ Q @ UA )
    # solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
    L1  = St @ UA.T + tau*( U1.T @ Q )
    # update S
    S1  = L1 @ U1
   
    return U1, S1

def kryl_expm(A,B,nkryl,dt=1.0):
    """
    Efficient computation of the action of the exponential propagator of A 
    over a time interval dt on a matrix B using the block arnoldi factorisation
    
    X = expm(dt*A) @ B

    Parameters
    ----------
    A : np.array (n x n)
        System matrix to be exponentiated
    B : np.array (n x p)    p << n
        Matrix to be propagated
    nkryl : int
        Number of steps in the block arnoldi factorisation
        NB: the block arnoldi factorisation is of size p*nkryl x p*nkryl
    dt : float (default is 1.0)
        Time interval for the exponential propagator

    Returns
    -------
    np.array (n x p)
        Best approximation of X = expm(dt*A) @ B in the Krylov subspace 
        K(A,B,nkryl)

    """
    try:
        rk = B.shape[1]
    except:
        rk = 1
    Qb,Rb = linalg.qr(B,mode='economic')
    Q, H = arn(A,Qb,nkryl)
    p = rk*nkryl
    return (Q[:,:p] @ linalg.expm(dt*H[:p,:p])[:,:rk]) @ Rb