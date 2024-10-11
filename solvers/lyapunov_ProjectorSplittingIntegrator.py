import numpy as np
import time
import sys

from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import lu_factor, lu_solve, solve, solve_lyapunov, svd, svdvals, qr, inv, norm, expm

sys.path.append('..')

from solvers.lyap_utils import idx_first_below, setup_IC, increase_rank, M_ForwardMap, G_ForwardMap

def LR_OSI_base(U0, S0, A, Q, Tend, dtaim, torder=1, verb=0):
    
    if verb > 0:
        print('\nLow-rank operator-splitting method for Differential Lyapunov equations.\n')
    
    nt    = int(np.ceil(Tend/dtaim))
    dt    = Tend/nt
    tspan = np.linspace(0, Tend, num=nt, endpoint=True)
    
    if torder == 1:
        exptA = expm(dt*A)
    elif torder == 2:
        exptA = expm(dt/2*A)
    else:
        print('torder >= 2 currently not implemented.')
        raise ValueError
        return
    
    etime = time.time()
    iprint = max(1,int(np.floor(nt/10)))
    for it in range(nt):
        if (it % iprint == 0 or it == nt-1) and verb > 0:
            print(f'  Step {it+1:5d}: t = {tspan[it]:6.3f}')
        U0, S0 = LR_OSI_step(U0, S0, A, Q, dt, exptA, torder, verb)
    if verb > 0:
        print(f'Elapsed time:   {time.time()-etime:6.3f} seconds.')

    return U0, S0

def LR_OSI_rk_base(U0, S0, A, Q,Tend, dtaim, torder=1, verb=0, tol=1e-6):
    
    if verb > 0:
        print('\nRank-adaptive low-rank operator-splitting method for Differential Lyapunov equations.\n')
    
    nt    = int(np.ceil(Tend/dtaim))
    dt    = Tend/nt
    tspan = np.linspace(0, Tend, num=nt, endpoint=True)
    
    if torder == 1:
        exptA = expm(dt*A)
    elif torder == 2:
        exptA = expm(dt/2*A)
    else:
        print('torder >= 2 currently not implemented.')
        raise ValueError
        return
    
    rk_red_lock = 0
    etime = time.time()
    iprint = max(1,int(np.floor(nt/10)))
    rkv = np.zeros((nt,))
    
    U, S = set_initial_rank(U0, S0, A, Q, dt, exptA, torder, verb, tol)

    for it in range(nt):
        if (it % iprint == 0 or it == nt-1) and verb > 0:
            print(f'  Step {it+1:5d}: t = {tspan[it]:6.3f}')
        U0, S0 = LR_OSI_rk_step(U0, S0, A, Q, dt, exptA, torder, rk_red_lock=rk_red_lock, verb=verb, tol=tol)
        rkv[it] = U0.shape[1]
        
    if verb > 0:
        print(f'Elapsed time:   {time.time()-etime:6.3f} seconds.')

    return U0, S0, rkv

def LR_OSI_test(A, Q, X0, Xrk, Xref, Tend, dtaim, rk, torder=1, verb=1):
    
    if verb > 0:
        print('\nLow-rank operator-splitting method for Differential Lyapunov equations.\n')
    
    eps = 1e-12
    N = X0.shape[0]
    # check rank of inhomogeneity
    Sq = svdvals(Q)
    rkq = sum(Sq > eps)
    if verb > 0:
        print(f'Numerical rank of inhomogeneity B  ({N:d}x{N:d}): {rkq:3d} (tol={eps:.2e})')
    
    # check rank of initial data
    U,S,_ = svd(X0, full_matrices=False)
    rk0 = sum(S > eps)
    if verb > 0:
        print(f'Numerical rank of initial data  X0 ({N:d}x{N:d}): {rk0:3d} (tol={eps:.2e})')
    
    if verb > 0:
        print('\nMode rank:')
        print(f'  Chosen rank: {rk:d}\n')
        
    S0 = np.zeros((rk,rk))
    Utmp = np.random.random_sample((N,rk))
    rkmin = min(rk,rk0)
    S0[:rkmin,:rkmin] = np.diag(S[:rkmin])
    Utmp[:,:rkmin] = U[:,:rkmin]
    U0, _ = qr(Utmp, mode='economic')
    
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
        print(f'Begin iteration:   0 --> {Tend:6.3f}      dt = {dt:.5e}')
    etime = time.time()
    dtprint = 1.0
    iprint = max(1, int(np.floor(dtprint/dt)))
    res_rk, res_rf, svals  = [], [], []
    for it in range(nt):
        if not iprint ==0 and (it % iprint == 0 or it == nt-1):
            X = U0 @ S0 @ U0.T
            U,S,_ = svd(S0)
            res1 = np.linalg.norm(X-Xrk)/N
            res2 = np.linalg.norm(X-Xref)/N
            svals.append(S)
            res_rk.append(res1)
            res_rf.append(res2)
            if verb > 0:
                print(f'  Step {it+1:5d}: t = {tspan[it]:6.3f}, err rk: {res1:.4e}, err ref: {res2:.4e}')
        
        U0, S0 = LR_OSI_step(U0, S0, A, Q, dt, exptA, torder, 0)
         
    etime = time.time() - etime
    if verb > 0:
        print(f'Elapsed time:   {etime:4.2f} seconds.')

    return U0, S0, np.array(svals), res_rk, res_rf

def LR_OSI_rk_test(A, Q, X0, Xrk, Xref, Tend, dtaim, torder=1, verb=1, tol=1e-6):
    
    if verb > 0:
        print('\nLow-rank operator-splitting method for Differential Lyapunov equations.\n')
    
    eps = 1e-12
    N = X0.shape[0]
    # check rank of inhomogeneity
    Uq,Sq,_ = svd(Q)
    rkq = sum(Sq > eps)
    if verb > 0:
        print(f'Numerical rank of inhomogeneity B  ({N:d}x{N:d}): {rkq:3d} (tol={eps:.2e})')
    
    # check rank of initial data
    U,S,_ = svd(X0, full_matrices=False)
    rk0 = sum(S > eps)
    if verb > 0:
        print(f'Numerical rank of initial data  X0 ({N:d}x{N:d}): {rk0:3d} (tol={eps:.2e})')
    
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
    
    U0, S0 = set_initial_rank(U, np.diag(S), A, Q, dt, exptA, torder, verb, tol=tol)
    
    if verb > 0:
        print(f'Begin iteration:   0 --> {Tend:6.3f}      dt = {dt:.5e}, tol_rk = {tol:.0e}')
    etime = time.time()
    dtprint = 0.1
    iprint = max(1, int(np.floor(dtprint/dt)))
    res_rk, res_rf, rkvec, svraw = [], [], [], []
    for it in range(nt):
        if not iprint ==0 and (it % iprint == 0 or it == nt-1):
            X = U0 @ S0 @ U0.T
            U,S,_ = svd(S0)
            svraw.append(S)
            res1 = np.linalg.norm(X-Xrk)/N
            res2 = np.linalg.norm(X-Xref)/N
            if verb > 0:
                print(f'  Step {it+1:5d}: t = {tspan[it]:6.3f}, rk: {U0.shape[1]-1}, err RK: {res1:.4e}, err ref: {res2:.4e}')
                #print(f'  svals: {svraw[-1]}')
            res_rk.append(res1)
            res_rf.append(res2)
            rkvec.append(U0.shape[1] - 1)
        
        U0, S0 = LR_OSI_rk_step(U0, S0, A, Q, dt, exptA, torder, 0, tol=tol)
         
    etime = time.time() - etime
    if verb > 0:
        print(f'Elapsed time:   {etime:4.2f} seconds.')
    
    tvec = np.linspace(0, Tend, len(rkvec))
    
    rkmax = max(rkvec) + 1
    svals = np.zeros((len(svraw), rkmax))
    for i, sv in enumerate(svraw):
        svals[i,:] = np.append(svraw[i], np.zeros(rkmax-len(sv)))
        
    return U0, S0, svals, res_rk, res_rf, rkvec, tvec

def LR_OSI_rk_step(U, S, A, Q, dt, exptA, torder, rk_red_lock, verb=0, rkmin=2, max_step=5, tol=1e-6):
    accept_step = False
    istep = 0 
    #print(f'rk init: {U.shape[1]}')
    while (not accept_step and istep < max_step):
        istep += 1
        n, rk = U.shape
        # regular step
        U, S = LR_OSI_step(U, S, A, Q, dt, exptA, torder, verb)
        _,svals,_ = svd(S)
        #print(svals)
        if svals[-1] > tol:
            # increase rank
            #print(f'{rk+1}')
            U, S = increase_rank(U, S)
            # avoid oscillations
            rk_red_lock = 10                
        else:
            # the maximum rank is at least sufficient
            accept_step = True
            idx = idx_first_below(svals, tol) + 1
            # check if we can reduce rank
            if not idx == rk and rk_red_lock == 0: # the rank is too large
                # decrease rank
                rknew = max(idx, rk-2) # reduce at most by 2
                if rknew >= rkmin:
                    S = S[:rknew,:rknew]
                    U = U[:, :rknew]
                    #print(f'New rank: {rknew-1:%d}')
                else:
                    #print('cannot reduce')
                    pass     
    if rk_red_lock > 0:
        rk_red_lock -= 1
    
    return U, S

def LR_OSI_step(U0, S0, A, Q, dt, exptA, torder, verb):
    
    etime = time.time()
    if torder == 1:
        U1A, S1A = M_ForwardMap(A, U0, S0, dt, exptA)
        U1, S1   = G_ForwardMap(U1A, S1A, Q, dt)
    elif torder == 2:
        U1A, S1A = M_ForwardMap(A, U0, S0, dt/2, exptA)
        U2, S2   = G_ForwardMap(U1A, S1A, Q, dt)
        U1, S1   = M_ForwardMap(A, U2, S2, dt/2, exptA)
    etime = time.time() - etime
    if verb > 0:
        print(f'Elapsed time:   {etime:4.2f} seconds.')
    return U1, S1

def set_initial_rank(U0, S0, A, Q, dt, exptA, torder, verb, tol, ninit=5, rkmin=1):
    accept_rank = False
    rk = rkmin
    n, rk0 = U0.shape
    #print(f'Initial condition: {rk0}')
    while not accept_rank:
        U, S = setup_IC(U0, S0, rk)
        Uout = U.copy()
        Sout = S.copy()
        for i in range(ninit):
            U, S = LR_OSI_step(U, S, A, Q, dt, exptA, torder, verb=0)
        _,svals,_ = svd(S)
        if svals[-1] > tol:
            # increase rank
            rk *= 2
            #print(f'{rk}')
        else:
            accept_rank = True
            rk = idx_first_below(svals, tol)
    print(f'accepted rank: {rk},  with  tol = {tol:.0e}')
    print(f' svals[{rk-1}:{rk}] = {svals[rk-1]} | {svals[rk]}')
    return Uout[:,:rk], Sout[:rk,:rk]
    