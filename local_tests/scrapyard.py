import numpy as np
import time
from scipy import linalg as LA
from scipy import optimize as Opt

# compute inner product of eigendirections
A = vs.conj().T @ vs
# compute reduced system matrix Lred
U,S,__ = LA.svd(A)
s = np.sqrt(S)
F = np.diag(s) @ U.conj().T
Finv = U @ np.diag(np.reciprocal(s))
Lred = (F @ np.diag(ds) @ Finv)

# compute exponential propagator for time Tmax
start = time.time()
Phi = LA.expm(Tmax*L)
end = time.time()
etime = end-start
print(etime)
# compute amplification
U,S,V = LA.svd(Phi)
sigma = S[0]**2

start = time.time()
PhiV = v @ np.diag(np.exp(d*Tmax)) @ LA.inv(v)
end = time.time()
etime = end-start

print(LA.norm(Phi - PhiV,ord=2))

start = time.time()
tmax1 = Opt.minimize_scalar(lambda t: -LA.norm(LA.expm(t*L)), bounds = [10, 20], method='bounded')
end = time.time()
etime1 = end-start
print(etime1)

start = time.time()
tmax2 = Opt.minimize_scalar(lambda t: -LA.norm(v @ np.diag(np.exp(t*d)) @ LA.inv(v)), bounds = [10, 20], method='bounded')
end = time.time()
etime2 = end-start

start = time.time()
tmax3 = Opt.minimize_scalar(lambda t: -LA.norm(vs @ np.diag(np.exp(t*ds)) @ LA.inv(vs)), bounds = [10, 20], method='bounded')
end = time.time()
etime3 = end-start
print()
print(etime1)
print(etime2)
print(etime3)
# print(tmax1)
# print(tmax2)

#print(f'Amplification at time T = {Tmax:.2f}: sigma = {sigma:.2f}\n')
"""Phi = LA.expm(Tmax*Lred)
 # compute amplification
U,S,V = LA.svd(Phi)
sigma = S[0]**2
print(f'Amplification at time T = {Tmax:.2f} (red): sigma = {sigma:.2f}\n')"""

# tv = np.linspace(0,100,100)

# sigma = np.ones(100,)
# for i,t in enumerate(tv[1:]):
#     sigma[i+1] = LA.norm(LA.expm(t*L))**2

# plt.figure()
# plt.plot(tv,sigma)
# plt.axvline(x=12.722961238252015,color='k')
# plt.show()

def lrcfadic_r2(A,B,pin,stop_criterion,criterion_type,Xref):
    
    if criterion_type == 'niter':
        res_tol = False    
        niter = stop_criterion
        tol = None
    elif criterion_type == 'tol':
        res_tol = True
        niter = 50
        tol = stop_criterion
    else:
        raise ValueError('Unsupported stop_criterion_type')
    
    nord = 'fro'
    I = np.eye(A.shape[0])
    
    pin, status, is_real = check_shifts(pin)
    
    etime = time.time()
    is_first = True
    lu_v = []
    p_v = np.array([])
    for (i, is_r) in enumerate(is_real):
        shift = pin[i]
        if is_r:
            if_add = True
            is_first = True
        else:
            if is_first:
                if_add = True
                is_first = False
            else:
                if_add = False
                is_first = True
        if if_add:
            print(shift)
            lu, piv = linalg.lu_factor(A + shift*I)
            lu_v.append((lu,piv))
            p_v = np.append(p_v,shift)

    # randomly permute array
    l = p_v.size
    #idx  = np.random.permutation(range(l))
    #p_v  = p_v[idx]
    #lu_v = [ lu_v[i] for i in idx ]
    print(f'Number of shifts: {pin.size}')
    print(f'Number of complex shifts: {sum(~is_real)}')
    print(f'Number of shifts for iteration: {l}')
        
    res_step = max(l,20)
    is_real = np.imag(p_v) == 0
            
    etime_LU = time.time() - etime
    
    Q,R  = linalg.qr(B,mode='economic')
            
    nrm0 = linalg.norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    #res_log = [ np.log10(nrm0) ]
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
    V_old       = B # dummy
    p_old       = 1 # dummy
    Z           = np.empty(A.shape[0],)
    
    for i in range(niter):
        etime = time.time()
        
        p = p_v[ip]
        
        if i == 0:
            q  = np.sqrt(-2*p)
            V  = q*linalg.lu_solve(lu_v[ip],B)
        else:
            Vt = linalg.lu_solve(lu_v[ip],V_old)
            qr = np.sqrt(np.real(p)/np.real(p_old))
            V  = qr*( V_old - (p + p_old.conj())*Vt )
        
        #print(f'Step {i+1}:  p = {p}\n  V : {V.dtype}')
        # update Cholesky factor

        if is_real[ip]:
            Z    = np.column_stack([ Z , np.real(V) ])
            Vb   = np.real(V)
        else: # double step
            beta = 2*np.real(p)/np.imag(p)
            V    = V.conj() + beta*np.imag(V)
            Z1   = np.sqrt(2)*( np.real(V) + 0.5*beta*np.imag(V) )
            Z2    = np.sqrt(0.5*beta**2 + 2)*np.imag(V)
            Vb   = np.column_stack([ Z1, Z2 ])
        Z = np.column_stack([ Z , Vb ])
        #print(f'  Z : {Z.dtype}')
            
        p_old = p
        V_old = V

        #print(f'  Old shift after step: pold = {pold}\n\n')      
        ip = (ip + 1) % l       # update shift index   
                    
        etime_CFADI = etime_CFADI + time.time() - etime
        
        resid = linalg.norm(Xref - Z @ Z.T, 'fro')
        print(f'Low-Rank CF-ADI, step {i}: res = {resid} (fro)')
          
        if if_res and i > 0 and ( i < 50 or i % res_step == 0):
            etime = time.time()
            ncols.append(V.shape[1])
            Q, R, nrm = residual(Vb, A, Q, R, ncols, nord)
            res.append(nrm)
            #res_log.append(np.log10(nrm))
            res_rel.append(nrm/nrm0)
            ires.append(i)
            #print(f'Low-Rank CF-ADI, step {i}: res = {res_rel[-1]}')
            
            nrmx.append(linalg.norm(Xref - Z @ Z.T, ord=nord)/nrmx0)
            nrmz.append(linalg.norm(Vb, ord=nord))
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
        raise ValueError('Unsupported stop_criterion_type')
    
    nord = 'fro'
    I = np.eye(A.shape[0])
    l = pin.size
    
    etime = time.time()
    is_real = np.isreal(pin)   
    lu_v = []
    p_v  = np.array([])
    print('Precompute LU decomposition:')
    for i, (p, is_r) in enumerate(zip(pin,is_real)):
        if is_r:
            Atmp = A + np.real(p)*I
            #print(f'  {i:3d}: p = {np.real(p):8.3f}      Atmp = A + pI')
        else:
            s_i = 2*np.real(p)
            t_i = np.abs(p)**2
            Atmp = A @ A + s_i*A + t_i*I
            #print(f'  {i:3d}: p = {p:8.3f}      Atmp = A^2 + 2*Re(p)*A + |p|^2*I')
        lu, piv = linalg.lu_factor(Atmp)
        lu_v.append((lu,piv))
        p_v = np.append(p_v,p)
        
    etime_LU = time.time() - etime
      
    res_step = max(l,20)   
    
    Q,R  = linalg.qr(B,mode='economic')
            
    nrm0 = linalg.norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    #res_log = [ np.log10(nrm0) ]
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
        q    = np.sqrt(-2*p1)
        V1t  = linalg.lu_solve(lu_v[0], B)
        Vnew = q*V1t
        V2t  = 0*V1t #dummy
    else:
        q1   = 2*np.sqrt(-np.real(p1))*np.abs(p1)
        q2   = 2*np.sqrt(-np.real(p1))
        V1t  = linalg.lu_solve(lu_v[0], B)
        V1   = q1*V1t
        V2t  = A @ V1t
        V2   = q2*V2t
        Vnew = np.column_stack([ V1, V2 ])
    Z = Vnew
    
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
                #print(f'\n{i}: p = {p:8.3f}          p real, p_old real')
                f1  = (p + p_old)
                V1t =   V1t_old \
                      - f1*linalg.lu_solve(lu_v[ip], V1t_old)
            else: # p_old complex
                #print(f'\n{i}: p = {p:8.3f}          p real, p_old complex')
                #print(f'p_old = {p_old:8.3f}')
                f1  = 2*np.real(p_old) + p
                f2  = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
                V1t =      V2t_old \
                      - f1*V1t_old \
                      + f2*linalg.lu_solve(lu_v[ip], V1t_old)
            Vnew = q*V1t
            V2t  = 0*V1t
        else: # p complex
            q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
            q2 = 2*np.sqrt(-np.real(p))
            if p_old_is_real:
                #print(f'\n{i}: p = {p:8.3f}          p complex, p_old real')
                Vtmp =     A @ V1t_old \
                       - p_old*V1t_old
                V1t  = linalg.lu_solve(lu_v[ip], Vtmp)
            else: # p_old complex
                #print(f'\n{i}: p = {p:8.3f}          p complex, p_old complex')
                f1   = np.abs(p_old)**2 - np.abs(p)**2
                f2   = 2*np.real(p_old + p)
                Vtmp =   f1*V1t_old \
                       - f2*V2t_old        
                V1t  =   V1t_old \
                       + linalg.lu_solve(lu_v[ip], Vtmp)
            V2t  = A @ V1t
            V1   = q1*V1t
            V2   = q2*V2t
            Vnew = np.column_stack([ V1, V2 ]) 
        Z = np.column_stack([ Z, Vnew ])
            
        p_old   = p
        V1t_old = V1t
        V2t_old = V2t
        #print(f'  Old shift after step: pold = {pold}\n\n')
                    
        etime_CFADI = etime_CFADI + time.time() - etime
        
        #resid = linalg.norm(Xref - Z @ Z.T, 'fro')
        #print(f'Low-Rank CF-ADI, step {i}: res = {resid} (fro)')
          
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

def lrcfadic(A,B,pin,stop_criterion,criterion_type,Xref):
    
    if criterion_type == 'niter':
        res_tol = False    
        niter = stop_criterion
        tol = None
    elif criterion_type == 'tol':
        res_tol = True
        niter = 10
        tol = stop_criterion
    else:
        raise ValueError('Unsupported stop_criterion_type')
    
    nord = 2 #'fro'
    I = np.eye(A.shape[0])
    is_real = np.imag(pin) == 0
    
    etime = time.time()
    lu_v = []
    is_first = True
    p     = np.array([])
    sigma = np.array([])
    tau   = np.array([])
    for (i, is_r) in enumerate(is_real):
        shift = pin[i]
        if is_r:
            lu, piv = linalg.lu_factor(A + shift*I)
            lu_v.append((lu,piv))
            p     = np.append(p,shift)
            sigma = np.append(sigma,0)
            tau   = np.append(tau,0)
        else:
            if is_first:
                s_i = 2*np.real(-shift)
                t_i = np.abs(shift)**2
                lu, piv = linalg.lu_factor(A @ A - s_i*A + t_i*I)
                lu_v.append((lu,piv))
                is_first = False
                p     = np.append(p,shift)
                sigma = np.append(sigma,s_i)
                tau   = np.append(tau,t_i)
            else: # add only one decomposition per cc pair
                is_first = True
          
    print(f'Number of shifts: {pin.size}')
    print(f'Number of complex shifts: {sum(~is_real)}')
    print(f'Number of shifts for iteration: {p.size}')
    p = np.random.permutation(p)
    
    #print(sigma)
    #print(tau)
    
    l = p.size
    res_step = max(l,20)
    is_real = np.imag(p) == 0
            
    etime_LU = time.time() - etime
    
    Q,R  = linalg.qr(B,mode='economic')
            
    nrm0 = linalg.norm(R @ R.T, ord=nord)

    ncols = [ B.shape[1] ]

    #res_log = [ np.log10(nrm0) ]
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
    is_first = True
    is_real_old = False
    pold = 0
    sold = 0
    told = 0
    
    etime_res   = []
    etime_CFADI = 0
    
    for i in range(niter):
        etime = time.time()
        
        p_is_real = is_real[ip]
        p_i = p[ip]
        #print(f'Step {i+1}: ip = {ip}')
        #print(f'  Current shift: {pi}')
        #print(f'  Old shift    : {pold}')
        if p_is_real:
            #print('  real')
            p_i = np.real(p_i)
            s_i = sigma[ip]
            t_i = tau[ip]
            if i == 0:
                #print(f'Step {i}: p_i = {p_i}: real. ip = {ip}')
                V = np.sqrt(-2*p_i)*linalg.lu_solve(lu_v[ip],B)
                Z = np.real(V)
            else:
                #print(f'Step {i}: p_i = {p_i}: real. real_old = {is_real_old}. ip = {ip}')
                f1 = np.sqrt(-2*p_i)/np.sqrt(-2*pold)
                if is_real_old:
                    f2 = np.real(p_i + pold)
                    V = f1*(V - f2*linalg.lu_solve(lu_v[ip],V))
                else:
                    V = (A @ A + s_i*A + t_i*I)*linalg.lu_solve(lu_v[ip],V)                     
                Z = np.column_stack([ Z , np.real(V) ])
            pold = p_i
            sold = s_i
            told = t_i
            is_real_old = True
        else:
            #print('  c.c')
            s_i = sigma[ip]
            t_i = tau[ip]
            if i == 0:
                #print(f'Step {i}: p_i = {p_i}: c.c. ip = {ip}')
                f1 = np.sqrt(-2*np.real(p_i))
                V  = f1*linalg.lu_solve(lu_v[ip],B)
                Z  = np.real(V)
                #print(f'f1 = {f1}')
            else:
                #print(f'Step {i}: p_i = {p_i}: c.c.  real_old = {is_real_old}. ip = {ip}')
                f1 = np.sqrt(-2*np.real(p_i))/np.sqrt(-2*pold)
                if is_real_old:
                    V = f1*(A @ A + s_i*A + t_i*I)*linalg.lu_solve(lu_v[ip],V) 
                else:
                    f2 = (s_i+ sold)*A + (t_i + told)*I
                    V = f1*(V - f2 @ linalg.lu_solve(lu_v[ip],V))
                print(f'f1 = {f1}')
                print(f' s_i/sold = {s_i}/{sold}')
                print(f' t_i/told = {t_i}/{told}')
                Z  = np.column_stack([ Z , np.real(V) ])
            pold = np.real(p_i)
            sold = s_i
            told = t_i
            is_real_old = False
        #print(f'  Old shift after step: pold = {pold}\n\n')      
        ip = (ip + 1) % l       # update shift index   
                    
        etime_CFADI = etime_CFADI + time.time() - etime
        
        resid = linalg.norm(Xref - Z @ Z.T, 'fro')
        print(f'Low-Rank CF-ADI, step {i}: res = {resid} (fro)')
          
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