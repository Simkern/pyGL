# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:32:04 2024

@author: Simon
"""

import numpy as np
from scipy.sparse.linalg import gmres, cg, LinearOperator
from scipy.linalg import lu_factor, lu_solve

def gmres_wrapper(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, callback=None):
    
    # Variables to store information
    residuals = []
    Axcount   = [0]

    # Define a callback function to collect information at each iteration
    def gmres_callback(rk):
        Axcount[0] += 1
        residuals.append(rk)

    # Call scipy's gmres function with the provided parameters
    x, info = gmres(A, b, x0=x0, tol=tol,
                    restart=restart, maxiter=maxiter,
                    callback=gmres_callback,
                    callback_type='pr_norm')

    return x, info, Axcount, residuals, len(residuals)

def pgmres_lu(A, b, L, x0=None, tol=1e-5, restart=None, maxiter=None, callback=None):
    
    lu,piv = lu_factor(L)
    
    # Variables to store information
    residuals = []
    Axcount   = [0]
    Pxcount   = [0]

    # Define preconditioner
    def Minv(v):
        Pxcount[0] += 1
        return lu_solve((lu,piv), v)

    # Define a callback function to collect information at each iteration
    def gmres_callback(rk):
        Axcount[0] += 1
        residuals.append(rk)
        
    opM = LinearOperator(matvec=Minv, shape = A.shape, dtype = A.dtype)

    # Call scipy's gmres function with the provided parameters
    x, info = gmres(A, b, M=opM, x0=x0, tol=tol,
                    restart=restart, maxiter=maxiter,
                    callback=gmres_callback,
                    callback_type='pr_norm')

    return x, info, Axcount, Pxcount, residuals, len(residuals)

def pgmres_cg(A, b, L, x0=None, tol=None, ptol=None, restart=None, maxiter=None, callback=None):
    
    # Variables to store information
    Pres = []
    res  = []
    Axcount    = [0]
    Pxcount    = [0]
    Mxcount    = []
    
    assert(np.all(np.linalg.eigvals(L) > 0))
    
    # Define preconditioner
    def Minv(v):
        Pxcount[0] += 1
        Mxcount.append(0)
        Pres.append([])
        x, info = cg(opM, v, tol=ptol, 
                     maxiter = 90,
                     callback = cg_callback)
        print(np.linalg.norm(L @ x - v))
        return x
    
    # Define a callback functiosn
    def gmres_callback(rk):
        Axcount[0] += 1
        res.append(rk)
        
    def cg_callback(xk):
        Mxcount[-1] += 1
        #print(b.shape)
        #print(np.linalg.norm(L @ xk - b))
        Pres[-1].append(np.linalg.norm(L @ xk - b))
    
    opM    = LinearOperator(matvec=lambda x: L @ x, shape=A.shape, dtype=A.dtype)
    opMinv = LinearOperator(matvec=Minv, shape=A.shape, dtype=A.dtype)
    
    # Call scipy's gmres function with the provided parameters
    x, info = gmres(A, b, M=opMinv, x0=x0, tol=tol,
                    restart=restart, maxiter=maxiter,
                    callback=gmres_callback,
                    callback_type='pr_norm')

    return x, info, Axcount, Pxcount, Mxcount, res, Pres


def cg_wrapper(L, b, x0=None, tol=1e-6, maxiter=None, callback=None):

    residuals = []
    Pxcounter = [0]
    
    def cg_callback(xk):
        Pxcounter[0] += 1
        residuals.append(np.linalg.norm(Mx(xk) - b))
        
    #def Mx(v):
    #    return (Lc @ v.reshape(Nxc, 2, order='F')).reshape(Nx, 1, order='F')
    def Mx(v):
        return L @ v
    
    opL = LinearOperator(matvec=Mx, shape=L.shape, dtype=L.dtype)
    
    x, info = cg(opL, b, x0=x0, tol=tol, 
                 maxiter=maxiter,
                 callback=cg_callback)
    
    print(np.linalg.norm(L @ x - b))
    
    return x, info, Pxcounter, residuals, len(residuals)
    
    