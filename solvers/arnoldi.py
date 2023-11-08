import numpy as np
import scipy
import sys

sys.path.append('..')

from core.utils import en

def arn(A, b, n):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.
    
    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    H : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    eps = 1e-12
    H = np.zeros((n + 1, n), dtype=complex)
    Q = np.zeros((A.shape[0], n + 1), dtype=complex)
    # Normalize the input vector
    Q[:, 0] = b / en(b)  # Use it as the first Krylov vector
    for k in range(n):
        v = A @ Q[:, k ]  # Generate a new candidate vector
        # for each colum of Q that we have already constructed
        for i in range(k+1):
            # compute projection
            H[i,k] = np.dot(Q[:,i].conj() , v)
            # project out direction
            v = v - H[i,k] * Q[:,i]
        # normalize
        beta = en(v)
        H[k+1, k] = beta
        # stopping criterion
        if H[k+1, k] > eps:
            Q[:, k+1] = v / beta
        else:
            return Q, H
    return Q, H

def arn_inv(A, b, n):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A^-1.

    This is the space spanned by the vectors {b, A^-1 b, ..., A^-n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
        The algorithm will compute the LU factorisation and solve the linear system with it at each iterate
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.
    
    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    H : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    eps = 1e-12
    H = np.zeros((n + 1, n), dtype=complex)
    Q = np.zeros((A.shape[0], n + 1), dtype=complex)
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
            v = v - H[i,k] * Q[:,i]
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
        z = z - proj[i] * Q[:,i]
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