#!/usr/bin/env python3
# --> Standard Python packages.
import numpy as np
import sys
import matplotlib.pyplot as plt

# --> SciPy built-in minimization function.
from scipy.optimize import minimize

import scipy.linalg as linalg
from scipy.integrate import solve_ivp

from scipy.linalg import solve_continuous_are as care
from scipy.linalg import solve_continuous_lyapunov as cale

# --> LTI system related utilities.
from scipy.signal import lti, dlti, dlsim

from core.utils import p

def Xdot(t,xv,A,B,Q,Rinv):
    X = np.reshape(xv, A.shape)
    return np.reshape(A.T @ X + X @ A + Q - X @ B @ Rinv @ B.T @ X, xv.shape)

def Xdot_L(t,xv,A):
    X = np.reshape(xv, A.shape)
    return np.reshape(A.T @ X + X @ A, xv.shape)

def OSI_step(U,S,A,B,Q,Rinv,tau,torder=1):
    if torder==1:
        UA, SA = M_ForwardMap(        U,  S,  A,             tau)
        U, S   = G_forwardMap_Riccati(UA, SA, A, B, Q, Rinv, tau)
    else:
        UA, SA = M_ForwardMap(        U,  S,  A,             tau/2)
        U1, S1 = G_forwardMap_Riccati_2(UA, SA, A, B, Q, Rinv, tau)
        U,  S  = M_ForwardMap(        U1, S1, A,             tau/2)
    return U, S

def M_ForwardMap(U,S,A,tau):
    U1    = linalg.expm(tau*A.T) @ U
    UA, R = linalg.qr(U1, mode='economic')
    SA    = R @ S @ R.T
    return UA, SA


def G_ForwardMap(UA, SA, Q, tau):
    # solve Kdot = Q @ UA with K0 = UA @ SA for one step tau
    K1 = UA @ SA + tau*(Q @ UA)
    # orthonormalise K1
    U1, Sh = linalg.qr(K1, mode='economic')
    # solve Sdot = - U1.T @ Q @ UA with S0 = Sh for one step tau
    St = Sh - tau*( U1.T @ Q @ UA )
    # solve Ldot = U1.T @ Q with L0 = St @ UA.T for one step tau
    L1  = St @ UA.T + tau*( U1.T @ Q )
    # update S
    S1  = L1 @ U1
    return U1, S1


def G_forwardMap_Riccati_py(UA, SA, A, B, Q, Rinv, tau):
    
    P = B @ Rinv @ B.T
    # K
    K0   = UA @ SA
    Kdot = Q @ UA - K0 @ UA.T @ P @ K0
    K1   = K0 + tau*Kdot
    U1, S1 = linalg.qr(K1, mode='economic')
    # S
    S0   = S1.copy()
    Sdot = -U1.T @ Q @ UA + S0 @ UA.T @ P @ U1 @ S0
    Sh   = S0 + tau*Sdot
    # L #L0   = Sh @ UA.T #Ldot = U1.T @ Q - L0 @ P @ U1 @ L0 #L1   = L0 + tau*Ldot #S1   = L1 @ U1
    # L.T
    L0T   = UA @ Sh.T
    LTdot = Q @ U1 - L0T @ U1.T @ P @ L0T
    L1T   = L0T + tau*LTdot
    S1    = L1T.T @ U1
    return U1, S1

def K_step_Riccati(U, S, A, B, Q, Rinv, tau, QU, UTB, reverse=False):
    
    if not reverse:
        # constant
        QU     = Q @ U
        # non-linear
        UTB    = U.T @ B
    U1     = U @ S    
    BTU    = B.T @ U
    Uwrk0  = U1 @ UTB @ Rinv @ BTU @ S
    # combine
    Uwrk0_ = QU - Uwrk0
    U1_    = U1 + tau*Uwrk0_
    
    U1, S1 = linalg.qr(U1_, mode='economic')
    
    return U1, S1, QU, UTB
    
def S_step_Riccati(U1, U, S, A, B, Q, Rinv, tau, QU, UTB, reverse=False):
    
    if reverse:
        # constant
        QU     = Q @ U
        # non-linear
        UTB    = U.T @ B
    
    Swrk0  = U1.T @ QU
    # non-linear
    BTU    = B.T @ U1
    Swrk1  = S @ UTB @ Rinv @ BTU @ S
    # combine
    Swrk0  = - Swrk0 + Swrk1
    S1 = S + tau*Swrk0
    
    return S1, QU, UTB
    
def L_step_Riccati(U1, U, S, A, B, Q, Rinv, tau):
    
    Uwrk0  = Q @ U1
    # non-linear
    Uwrk1  = U @ S.T
    UTB    = U1.T @ B
    BTU    = B.T @ Uwrk1
    Swrk0  = UTB @ Rinv @ BTU
    U      = Uwrk1 @ Swrk0
    # combine
    Uwrk0_= Uwrk0 - U
    Uwrk1_= Uwrk1 + tau*Uwrk0_
    return Uwrk1_.T @ U1    

def G_forwardMap_Riccati(UA, SA, A, B, Q, Rinv, tau):
    
    U1, S1, QU, UTB = K_step_Riccati(    UA, SA, A, B, Q, Rinv, tau, None, None)
    Sh    , QU, UTB = S_step_Riccati(U1, UA, S1, A, B, Q, Rinv, tau, QU, UTB)
    S1              = L_step_Riccati(U1, UA, Sh, A, B, Q, Rinv, tau)
        
    return U1, S1

def G_forwardMap_Riccati_2(UA, SA, A, B, Q, Rinv, tau):
    
    # prediction
    U0 = UA.copy()
    K0 = U0 @ SA
    T0 = K0 @ U0.T @ B @ Rinv @ B.T @ K0
    U1, S1 = K_step_Riccati_2(    UA, SA, A, B, Q, Rinv, tau, T0, False)
    Sh     = S_step_Riccati_2(U1, UA, S1, A, B, Q, Rinv, tau, T0, False)
    S1     = L_step_Riccati_2(U1, UA, Sh, A, B, Q, Rinv, tau)
    Ut = U1.copy()
    Kt = Ut @ S1
    Tt = Kt @ U1.T @ B @ Rinv @ B.T @ Kt
    # second order
    Z      = np.zeros_like(T0)
    U1, S1 = K_step_Riccati_2(    UA, SA, A, B, Q, Rinv, tau/2, T0)
    Sh     = S_step_Riccati_2(U1, UA, S1, A, B, Q, Rinv, tau/2, Z)
    S1     = L_step_Riccati_2(U1, UA, Sh, A, B, Q, Rinv, tau)
    Gamma = 0.5*(T0 @ U0.T @ U1 + Tt @ Ut.T @ U1)
    Sh     = S_step_Riccati_2(U1, U1, S1, A, B, Q, Rinv, tau/2, Gamma, True)
    U1, S1 = K_step_Riccati_2(    U1, Sh, A, B, Q, Rinv, tau/2, Gamma, True)
        
    return U1, S1

def K_step_Riccati_2(U, S, A, B, Q, Rinv, tau, T0, reverse=False):
    
    U1     = U @ S
    
    # constant
    QU     = Q @ U
    
    # combine
    Uwrk0 = QU - T0
    U1_   = U1 + tau*Uwrk0
    
    U1, S1 = linalg.qr(U1_, mode='economic')
    
    return U1, S1
    
def S_step_Riccati_2(U1, U, S, A, B, Q, Rinv, tau, T0, reverse=False):
    
    # constant
    QU     = Q @ U
    Swrk0  = U1.T @ QU
    
    # non-linear
    if not reverse:
        UTB    = U.T @ B
        BTU    = B.T @ U1
        Swrk1  = S @ UTB @ Rinv @ BTU @ S
    else:
        Swrk1  = U1.T @ T0
    # combine
    Swrk0  = - Swrk0 + Swrk1
    S1 = S + tau*Swrk0
    
    return S1
    
def L_step_Riccati_2(U1, U, S, A, B, Q, Rinv, tau):
    
    Uwrk0  = Q @ U1
    # non-linear
    Uwrk1  = U @ S.T
    UTB    = U1.T @ B
    BTU    = B.T @ Uwrk1
    Swrk0  = UTB @ Rinv @ BTU
    U      = Uwrk1 @ Swrk0
    # combine
    Uwrk0_= Uwrk0 - U
    Uwrk1_= Uwrk1 + tau*Uwrk0_
    return Uwrk1_.T @ U1    

if __name__ == "__main__":
    # --> Parameters of the system.
    ## parameters
    eps    = 1e-12
    n      = 4            # number of points per dimension
    rk_b   = 1             # rank of the RHS inhomogeneity
    rk_c   = 1             # rank of the RHS inhomogeneity
    rk_X0  = 1             # rank of the initial condition

    I   = np.eye(n)
    h   = 1/n**2
    At  = np.diag(-2/h*np.ones((n,)),0) + \
          np.diag(np.ones((n-1,))/h,1) + \
          np.diag(np.ones((n-1,))/h,-1)
    A   = np.kron(At,I) + np.kron(I,At)
    N   = A.shape[0]
    
    # Generate initial condition
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((N, rk_X0)),mode='economic')
    S0    = np.diag(s0);
    X0    = U0 @ S0 @ U0.T
    
    X0    = U0 @ S0 @ U0.T

    X = X0.copy()
    nrep = 6
    Tend = 0.2
    
    ###############################
    #
    #   Xdot = A.T X + X A
    #
    ###############################
    
    tspan = (0,Tend)
    print('\nXdot = A.T X + X A')
    print('\n  Runge-Kutta:\t\t||X||_2/N\t\t\t\t\tTend')
    print('\n\tX0:\t\t\t\t', end=' ')
    print(f'{np.linalg.norm(X0)/n:8.2e}\t\t\t\t\t{0:4.1f}')
    for irep in range(nrep):
        sol = solve_ivp(Xdot_L,tspan,X.flatten(),args=(A,), atol=1e-12, rtol=1e-12)
        X = sol.y[:,-1].reshape(A.shape)
        print(f'\tirep = {irep+1:2d}:', end='\t\t ')
        print(f'{np.linalg.norm(X)/n:8.2e} ', end='\t\t\t\t\t')
        print(f'{(irep+1)*Tend:4.1f}')
    
    dt = 0.001
    print('\n  Low-rank OS:\t\t||X||_2/N\t||X-X_RK||_2/N\tTend ',end='')
    print(f'(dt = {dt:8.6f})\n')
    U = U0.copy()
    S = S0.copy()
    XRK = sol.y[:,-1].reshape(A.shape)
    nsteps = int(np.ceil(Tend/dt))
    print('\tX0:\t\t\t\t', end=' ')
    print(f'{np.linalg.norm(X0)/n:8.2e}', end='\t\t  ')
    print(f'{np.linalg.norm(X0-XRK)/n:8.2e}', end='\t')
    print(f'{0:4.1f}')
    for irep in range(nrep):
        for i in range(nsteps):
            U1, S1 = M_ForwardMap(U, S,   A,             dt)
            U = U1.copy()
            S = S1.copy()
        X1 = U @ S @ U.T
        print(f'\tnsteps = {nsteps:5d}\t', end=' ')
        print(f'{np.linalg.norm(X1)/n:8.2e} ', end='\t\t  ')
        print(f'{np.linalg.norm(X1-XRK)/n:8.2e} ', end='\t')
        print(f'{(irep+1)*Tend:4.1f}')
    
    ###############################
    #
    #   Xdot = A.T X + X A + Q
    #
    ###############################
    Rinv = np.zeros((rk_b,rk_b))
    
    # Generate RHS
    CT = np.random.random_sample((rk_c,N))
    Qc = np.eye(rk_c)
    Q = CT.T @ Qc @ CT
    P0 = cale(A.T, -Q)
    
    X = X0.copy()
    nrep = 10
    Tend = 0.1
    tspan = (0,Tend)
    print('\nXdot = A.T X + X A + Q')
    
    print('\n  Runge-Kutta:\t  ||X-P0||_2/N\t\t\t\t\tTend')
    print('\n\tX0:\t\t\t\t', end=' ')
    print(f'{np.linalg.norm(X0)/n:8.2e}\t\t\t\t\t{0:4.1f}')
    for irep in range(nrep):
        sol = solve_ivp(Xdot,tspan,X.flatten(),args=(A,CT.T,Q,Rinv), atol=1e-12, rtol=1e-12)
        X = sol.y[:,-1].reshape(A.shape)
        print(f'\tirep = {irep+1:2d}:', end='\t\t ')
        print(f'{np.linalg.norm(X-P0)/n:8.2e} ', end='\t\t\t\t\t')
        print(f'{(irep+1)*Tend:4.1f}',end='\t')
        if irep == 0:
            print('\t<--')
        else:
            print('')
    Tend = 0.1
    tspan = (0,Tend)
    X = X0.copy()
    sol = solve_ivp(Xdot,tspan,X.flatten(),args=(A,CT.T,Q,Rinv), atol=1e-12, rtol=1e-12)
    XRK = sol.y[:,-1].reshape(A.shape)
    rkv = [2, 10]
    print('\n  Low-rank OS:\t ||X-P0||_2/N\t||X-X_RK||_2/N\tTend\t\t\tdt')
    for rk in rkv:
        print(f'rk = {rk:2d}')
        dtv = np.logspace(-5, -1, 5)
        for it, dt in enumerate(dtv[::-1]):
            U = np.zeros((N,rk))
            S = np.zeros((rk,rk))
            rkm = min(rk, rk_X0)
            U[:,:rkm] = U0[:,:rkm].copy()
            S[:rkm,:rkm] = S0[:rkm,:rkm].copy()
            nsteps = int(np.ceil(Tend/dt))
            for i in range(nsteps):
                UA, SA = M_ForwardMap(U,  S,  A, dt)
                U, S   = G_ForwardMap(UA, SA, Q, dt)
            X1 = U @ S @ U.T
            #
            print(f'\tnsteps = {nsteps:5d}\t', end=' ')
            print(f'{np.linalg.norm(X1-P0)/n:8.2e} ', end='\t\t  ')
            print(f'{np.linalg.norm(X1-XRK)/n:8.2e} ', end='\t')
            print(f'{Tend:4.1f}   {dt:8.6f}')
    ###############################
    #
    #   Xdot = A.T X + X A + Q - X B R^(-1) B.T X
    #
    ###############################
    R = 1e-1 * np.eye(rk_b)
    Rinv = np.linalg.inv(R)
    
    B = np.random.random_sample((N,rk_b))
    
    P  = care(A, B, Q, R)
    
    X = X0.copy()
    nrep = 5
    Tend = 0.1
    tspan = (0,Tend)
    print('\nXdot = A.T X + X A + Q - X @ B @ Rinv @ B.T @ X')
    
    print('\n  Runge-Kutta:\t  ||X-P||_2/N\t\t\t\t\t\t\t\tTend')
    print('\n\tX0:\t\t\t\t', end=' ')
    print(f'{np.linalg.norm(X0)/n:8.2e}\t\t\t\t\t\t\t\t{0:4.1f}')
    for irep in range(nrep):
        sol = solve_ivp(Xdot,tspan,X.flatten(),args=(A,B,Q,Rinv), atol=1e-12, rtol=1e-12)
        X = sol.y[:,-1].reshape(A.shape)
        print(f'\tirep = {irep+1:2d}:', end='\t\t ')
        print(f'{np.linalg.norm(X-P)/n:8.2e} ', end='\t\t\t\t\t\t\t\t')
        print(f'{(irep+1)*Tend:4.1f}',end='\t')
        if irep == 0:
            print('\t<--')
        else:
            print('')
    Tend = 0.1
    tspan = (0,Tend)
    X = X0.copy()
    sol = solve_ivp(Xdot,tspan,X.flatten(),args=(A,B,Q,Rinv), atol=1e-12, rtol=1e-12)
    XRK = sol.y[:,-1].reshape(A.shape)
    print('\n  Low-rank OS:\t  ||X-P||_2/N ||X_py-X_RK||_2/N ||X-X_RK||_2/N\tTend\t\t\tdt\n')
    dtv = np.logspace(-5, -1, 5)
    for it, dt in enumerate(dtv[::-1]):
        nsteps = int(np.ceil(Tend/dt))
        Upy = U0.copy()
        Spy = S0.copy()
        for i in range(nsteps):
            UA, SA     = M_ForwardMap(           Upy, Spy, A,             dt)
            Upy, Spy   = G_forwardMap_Riccati_py(UA,  SA,  A, B, Q, Rinv, dt)
        X1py = Upy @ Spy @ Upy.T
        U = U0.copy()
        S = S0.copy()
        for i in range(nsteps):
            UA, SA = M_ForwardMap(        U,  S,  A,             dt)
            U, S   = G_forwardMap_Riccati(UA, SA, A, B, Q, Rinv, dt)
        X1 = U @ S @ U.T
        #
        print(f'\tnsteps = {nsteps:5d}\t', end=' ')
        print(f'{np.linalg.norm(X1-P)/n:8.2e} ', end='\t\t  ')
        print(f'{np.linalg.norm(X1py-XRK)/n:8.2e} ', end='\t\t')
        print(f'{np.linalg.norm(X1-XRK)/n:8.2e} ', end='')
        print(f'{Tend:4.1f}   {dt:8.6f}')
       
    print('\n  Low-rank OS:\t\t||X-X_RK||_2/N (1)\t||X-X_RK||_2/N (2)\tTend\t\t\tdt\n')
    for it, dt in enumerate(dtv[::-1]):
        nsteps = int(np.ceil(Tend/dt))
        # torder 1
        U = U0.copy()
        S = S0.copy()
        for i in range(nsteps):
            U, S = OSI_step(U,S,A,B,Q,Rinv,dt,torder=1)
        X1 = U @ S @ U.T
        # torder 2
        U = U0.copy()
        S = S0.copy()
        for i in range(nsteps):
            U, S = OSI_step(U,S,A,B,Q,Rinv,dt,torder=2)
        X2 = U @ S @ U.T
        #
        print(f'\tnsteps = {nsteps:5d}\t', end='\t  ')
        print(f'{np.linalg.norm(X1-XRK)/n:8.2e} (1)', end='\t\t  ')
        print(f'{np.linalg.norm(X2-XRK)/n:8.2e} (2)', end='\t')
        print(f'{Tend:4.1f}   {dt:8.6f}')