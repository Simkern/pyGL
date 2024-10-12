import numpy as np
import scipy
import sys

sys.path.append('..')

from core.utils import en
from scipy import linalg

def arn(A,B,n, verb = 0):
    try:
        p = B.shape[1]
        if p == 1:
            Q, H = s_arn(A, B.flatten(), n, verb)
        else:
            Q, H = block_arn(A, B, n, verb)
    except:
        Q, H = s_arn(A, B, n, verb)
    return Q, H

def s_arn(A, b, n, verb = 0):
    eps = 1e-12
    H = np.zeros((n + 1, n), dtype=A.dtype)
    Q = np.zeros((A.shape[0], n + 1), dtype=A.dtype)
    # Normalize the input vector
    Q[:, 0] = b / en(b)  # Use it as the first Krylov vector
    for k in range(n):
        if verb > 0:
            print(f'  {k:3d}')
        v = A @ Q[:, k ]  # Generate a new candidate vector
        # for each colum of Q that we have already constructed
        for i in range(k+1):
            # compute projection
            H[i,k] = np.dot(Q[:,i].conj() , v)
            # project out direction
            v -= H[i,k] * Q[:,i]
        # normalize
        beta = en(v)
        H[k+1, k] = beta
        # stopping criterion
        if H[k+1, k] > eps:
            Q[:, k+1] = v / beta
        else:
            return Q, H
    return Q, H

def block_arn(A, B, n, verb = 0):
    #eps = 1e-12
    p = B.shape[1]
    H = np.zeros((p*(n+1), p*n), dtype=A.dtype)
    Q = np.zeros((A.shape[0], p*(n+1)), dtype=A.dtype)
    # Ortho-normalize the input matrix
    Q[:, :p], _ = scipy.linalg.qr(B, mode='economic')  # Use it as the first Krylov basis
    for k in range(n):
        if verb > 0:
            print(f'  {k:3d}')
        s   = slice(p*k,p*(k+1))
        sp1 = slice(p*(k+1),p*(k+2))
        V = A @ Q[:,s]  # Generate a new candidate matrix
        # for each colum of Q that we have already constructed
        for l in range(2):      # MGS
            k_min = max(0, k - n - 1)
            for ll in range(k_min, k + 1):
                sl       = slice(p*ll,p*(ll+1))
                proj     = Q[:,sl].conj().T @ V
                H[sl,s] += proj
                V       -= Q[:,sl] @ proj
        # Ortho-normalize result
        Q[:,sp1], H[sp1,s] = scipy.linalg.qr(V, mode='economic')
    return Q, H

def arn_inv(A, b, n):
    eps = 1e-12
    H = np.zeros((n + 1, n), dtype=A.dtype)
    Q = np.zeros((A.shape[0], n + 1), dtype=A.dtype)
    # Normalize the input vector
    Q[:, 0] = b / en(b)  # Use it as the first Krylov vector
    # Generate LU factorisation of A
    lu, piv = scipy.linalg.lu_factor(A)
    for k in range(n):
        v = scipy.linalg.lu_solve((lu, piv), Q[:, k])  # Generate a new candidate vector
        # for each colum of Q that we have already constructed
        for i in range(k+1):
            # compute projection
            H[i,k] = np.dot(Q[:,i].conj() , v)
            # project out direction
            v += - H[i,k] * Q[:,i]
        # normalize
        beta = en(v)
        H[k+1, k] = beta
        # stopping criterion
        if H[k+1, k] > eps:
            Q[:, k+1] = v / beta
        else:
            return Q, H
    return Q, H

def GS(Q,w,k,v):
    if v == 1:
        h,beta,z = CGS(Q,w,k)
    elif v == 2:
        h,beta,z = MGS(Q,w,k)
    elif v == 3:
        h,beta,z = DCGS(Q,w,k)
    return h,beta,z

def CGS(Q,w,k):
    proj = np.dot(Q.conj() , w)
    z = w - np.dot(Q , proj)
    beta = en(z)
    proj = proj[1:k]
    return proj,beta,z

def MGS(Q,w,k):
    proj = np.zeros((k,1))
    z = w
    # for each colum
    for i in range(k):
        proj[i] = np.dot(Q[:,i].conj() , w)
        # project out direction
        z += - proj[i] * Q[:,i]
    # normalize
    beta = en(z)
    return proj,beta,z

def DCGS(Q,w,k):
    p1,beta,z = CGS(Q,w,k)
    p2,beta,z = CGS(Q,z,k)
    proj = p1 + p2
    beta = en(z)
    proj = proj[1:k]
    return proj,beta,z

def kryl_expm(A,B,nkryl,dt=1.0,rho=None):
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
    Q_, H_ = arn(A,Qb,nkryl)
        
    kp  = rk*nkryl
    kpm = kp - rk
    kpp = kp + rk
    H   = H_[:kp,:kp]
    
    em = np.zeros((kp,rk))
    em[kpm:kp,:rk] = np.eye(rk)
    hemT = H_[kp:kpp,kpm:kp] @ em.T
    
    Hmh = np.block([[ H,    np.zeros((kp,rk)) ],
                    [ hemT, np.zeros((rk,rk)) ]])
                    
    expH = linalg.expm(dt*Hmh)
    
    y    = expH[:kpp,:rk] @ Rb
    x    = Q_ @ y
    
    #err = np.linalg.norm(abs(expH[kp:kpp,:rk] @ Rb))
    #print(err)
    
    return  x