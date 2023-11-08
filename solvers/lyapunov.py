import numpy as np
import time
import sys

from scipy import linalg

sys.path.append('..')

from solvers.lyap_utils import residual

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
    
    nord = 2 #'fro'
    l = p.size
    res_step = max(l,20)
    I = np.eye(A.shape[0])
    
    etime = time.time()
    lu_v = []
    for i in range(l):
        lu, piv = linalg.lu_factor(A + p[i]*I)
        lu_v.append((lu,piv))
    etime_LU = time.time() - etime

    Q,R  = linalg.qr(B,mode='economic')
            
    nrm0 = linalg.norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = linalg.norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ linalg.norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    
    etime_res   = []
    etime_CFADI = 0
    
    for i in range(niter):
        etime = time.time()
        if i == 0:
            V = np.sqrt(-2*p[ip])*linalg.lu_solve(lu_v[ip],B)
            Z = V
        else:
            pold = p[ip]        # keep old shift
            ip = (ip + 1) % l   # update shift index
        
            f1 = np.sqrt(-p[ip])/np.sqrt(-pold)
            f2 = np.real(p[ip] + pold)
             
            V = f1*(V - f2*linalg.lu_solve(lu_v[ip],V))
            
            Z = np.column_stack([ Z , V ])
        etime_CFADI = etime_CFADI + time.time() - etime
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            ncols.append(V.shape[1])
            Q, R, nrm = residual(V, A, Q, R, ncols, nord)
            res.append(nrm)
            #res_log.append(np.log10(nrm))
            res_rel.append(nrm/nrm0)
            ires.append(i)
            #print(f'Low-Rank CF-ADI, step {i}: res = {res_rel[-1]}')
            
            nrmx.append(linalg.norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(linalg.norm(V, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            #print(f'  Step {i+1:4d}: {nrmx[-1]:.9f}')
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_2/||X_ref||_2 = {nrmx[-1]}')
        print(f'  etime LU:    {etime_LU:8.6f} s')
        print(f'  etime solve: {etime_CFADI:8.6f} s')
        print(f'  etime res:   {np.sum(etime_res):8.6f} s')
        print(f'      min : {min(etime_res):8.6f} s')
        print(f'      max : {max(etime_res):8.6f} s')
        print(f'      avg : {np.mean(etime_res):8.6f} s\n')
    else:
        print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_2/||X_ref||_2 = {nrmx[-1]}')
        print(f'  etime LU:    {etime_LU:8.6f} s')
        print(f'  etime solve: {etime_CFADI:8.6f} s')
        print(f'  etime res:   {np.sum(etime_res):8.6f} s')
        print(f'      min : {min(etime_res):8.6f} s')
        print(f'      max : {max(etime_res):8.6f} s')
        print(f'      avg : {np.mean(etime_res):8.6f} s\n')
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel

def lrcfadic_r(A,B,pin,stop_criterion,criterion_type,Xref):
    
    if criterion_type == 'niter':
        res_tol = False    
        niter = stop_criterion
        tol = None
    elif criterion_type == 'tol':
        res_tol = True
        niter = 60
        tol = stop_criterion
    else:
        raise ValueError('Unsupported stop criterion')
    
    nord = 'fro'
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
        lu, piv = linalg.lu_factor(Atmp)
        lu_v.append((lu,piv))
        p_v = np.append(p_v,p)
        
    etime_LU = time.time() - etime
      
    res_step = max(l,20)   
    
    Q,R  = linalg.qr(B,mode='economic')
            
    nrm0 = linalg.norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    res     = [ nrm0 ]
    res_rel = [ 1 ] 
    if_res = True
    nrmx0 = linalg.norm(Xref, ord=nord)
    nrmx = [ 1 ]
    nrmz = [ linalg.norm(B, ord=nord) ]
    nrmz_rel = [ 1 ]

    ip = 0
    ires = [ 0 ]
    
    is_converged = False
    
    etime_res   = []
    etime_CFADI = 0
    Z           = np.empty(A.shape[0],)
    
    # initialisation
    p1         = p_v[0]
    p1_is_real = np.isreal(p1)
    if p1_is_real:
        p1 = np.real(p1)
        
    if p1_is_real:
        q   = np.sqrt(-2*p1)
        V1t = linalg.lu_solve(lu_v[0], B)
        V2t = 0*V1t #dummy
        Z   = q*V1t
    else: # p1 complex
        q1  = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2  = 2*np.sqrt(-np.real(p1))
        V1t = linalg.lu_solve(lu_v[0], B)
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
                V1t = V1t_old - f1*linalg.lu_solve(lu_v[ip], V1t_old)
            else: # p_old complex
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                V1t = V2t_old - f1*V1t_old + f2*linalg.lu_solve(lu_v[ip], V1t_old)
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                Vtmp = A @ V1t_old - p_old*V1t_old
                V1t  = linalg.lu_solve(lu_v[ip], Vtmp)
            else: # p_old complex
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp = f1*V1t_old - f2*V2t_old
                V1t  = V1t_old + linalg.lu_solve(lu_v[ip], Vtmp)
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
            ncols.append(Vnew.shape[1])
            Q, R, nrm = residual(Vnew, A, Q, R, ncols, nord)
            res.append(nrm)
            res_rel.append(nrm/nrm0)
            ires.append(i)
            nrmx.append(linalg.norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(linalg.norm(Vnew, ord=nord))
            nrmz_rel.append(nrmz[-1]/res[-2])
            etime_res.append(time.time() - etime)
            
            if res_tol and nrmx[-1] < tol:
                is_converged = True
                break
    etime_CFADI = time.time() - etime
            
    print('Low-Rank CF-ADI:')
    if is_converged:
        print(f'  Converged at step {i+1}.')
        print(f'  ||X_i - X_ref||_2/||X_ref||_2 = {nrmx[-1]}')
        print(f'  etime LU:    {etime_LU:8.6f} s')
        print(f'  etime solve: {etime_CFADI:8.6f} s')
        print(f'  etime res:   {np.sum(etime_res):8.6f} s')
        print(f'      min : {min(etime_res):8.6f} s')
        print(f'      max : {max(etime_res):8.6f} s')
        print(f'      avg : {np.mean(etime_res):8.6f} s\n')
    else:
        print(f'  Maximum number of iterations ({niter:d}) reached.')
        print(f'  ||X_i - X_ref||_2/||X_ref||_2 = {nrmx[-1]}')
        print(f'  etime LU:    {etime_LU:8.6f} s')
        print(f'  etime solve: {etime_CFADI:8.6f} s')
        print(f'  etime res:   {np.sum(etime_res):8.6f} s')
        print(f'      min : {min(etime_res):8.6f} s')
        print(f'      max : {max(etime_res):8.6f} s')
        print(f'      avg : {np.mean(etime_res):8.6f} s\n')
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel