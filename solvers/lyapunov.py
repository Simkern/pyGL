import numpy as np
import time
import sys

from itertools import compress
import matplotlib.pyplot as plt

from scipy.sparse.linalg import gmres
from scipy.linalg import lu_factor, lu_solve, qr, norm

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
        niter = 60
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
        etime = time.time()
        V1t, exitCode = gmres(A_v[0], B, tol=stol)
        etime_gmres[0] = time.time() - etime
        V2t = 0*V1t #dummy
        Z   = q*V1t
    else: # p1 complex
        q1  = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2  = 2*np.sqrt(-np.real(p1))
        etime = time.time()
        V1t, exitCode = gmres(A_v[0], B, tol=stol)
        etime_gmres[0] = time.time() - etime
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
                etime = time.time()
                Vtmp, exitCode = gmres(A_v[ip], V1t_old, tol=stol)
                etime_gmres[i] = time.time() - etime
                V1t = V1t_old - f1*Vtmp
            else: # p_old complex
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                etime = time.time()
                Vtmp, exitCode = gmres(A_v[ip], V1t_old, tol=stol)
                etime_gmres[i] = time.time() - etime
                V1t = V2t_old - f1*V1t_old + f2*Vtmp
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                Vtmp = A @ V1t_old - p_old*V1t_old
                etime = time.time()
                V1t, exitCode = gmres(A_v[ip], Vtmp, tol=stol)
                etime_gmres[i] = time.time() - etime              
            else: # p_old complex
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp = f1*V1t_old - f2*V2t_old
                etime = time.time()
                Vtmp2, exitCode = gmres(A_v[ip], Vtmp, tol=stol)
                etime_gmres[i] = time.time() - etime
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
    
    # fig = plt.figure(1)
    # etime_g      = etime_gmres.reshape((l,rounds))    
    # for i in range(rounds):
    #     if not np.sum(etime_g[:,i]) == 0:
    #         plt.plot(etime_g)
    # plt.yscale('log')
    # plt.show()
    
    
    return Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel