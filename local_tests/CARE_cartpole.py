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

# --> LTI system related utilities.
from scipy.signal import lti, dlti, dlsim

from core.utils import p
from CARE_Laplace import *

if __name__ == "__main__":
    # --> Parameters of the system.
    params = m, M, L, g, δ = 1, 5, 2, -10, 1
    n = 4
    p = 1
    q = 1
    rk_X0 = 2
    rk_rhs = 1
    
    A = np.array([
         [0.0, 1.0, 0.0, 0.0],                # Equation for the velocity dx/dt = ẋ
         [0, -δ/M, m*g/M, 0.0],               # Equation for the cart's acceleration.
         [0.0, 0.0, 0.0, 1.0],                # Equation for the pendulum angular velocity dθ/dt = θ̇
         [0.0, -δ/(M*L), -(m+M)*g/(M*L), 0.0] # Equation for the pendulum angular acceleration.
     ])
    
    # Generate initial condition
    s0    = np.random.random_sample((rk_X0,))
    U0, _ = linalg.qr(np.random.random_sample((n, rk_X0)),mode='economic')
    S0    = np.diag(s0);
    X0    = U0 @ S0 @ U0.T
    
    X0 = U0 @ S0 @ U0.T

    X = X0.copy()
    nrep = 6
    Tend = 0.2

    ###############################
    #
    #   Xdot = A.T X + X A
    #
    ###############################
    
    ###############################
    #
    #   Xdot = A.T X + X A + Q
    #
    ###############################
    Rinv = np.zeros((rk_rhs,rk_rhs))
    
    # Generate RHS
    C = np.random.random_sample((n,rk_rhs))
    C = C/np.linalg.norm(C)
    Q = C @ C.T
    #P0 = cale(A.T, -Q)
    
    ###############################
    #
    #   Xdot = A.T X + X A + Q - X B R^(-1) B.T X
    #
    ###############################
    R = 1e-1 * np.eye(rk_rhs)
    Rinv = np.linalg.inv(R)
    
    B = np.random.random_sample((n,rk_rhs))
    
    P  = care(A, B, Q, R)
   
    X = X0.copy()
    nrep = 5
    Tend = 5.0
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
    
    X = X0.copy()
    sol = solve_ivp(Xdot,tspan,X.flatten(),args=(A,B,Q,Rinv), atol=1e-12, rtol=1e-12)
    XRK = sol.y[:,-1].reshape(A.shape)
    print('\n  Low-rank OS:\t  ||X-P||_2/N ||X_py-X_RK||_2/N ||X-X_RK||_2/N\tTend\t\t\tdt\n')
    dtv = np.logspace(-4, -1, 4)
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
        U = U0.copy()
        S = S0.copy()
        for i in range(nsteps):
            UA, SA = M_ForwardMap(           U,  S,  A,             dt)
            U, S   = G_forwardMap_Riccati_py(UA, SA, A, B, Q, Rinv, dt)
        X1 = U @ S @ U.T
        U = U0.copy()
        S = S0.copy()
        for i in range(nsteps):
            UA, SA = M_ForwardMap(        U,  S,  A,             dt/2)
            U1, S1 = G_forwardMap_Riccati(UA, SA, A, B, Q, Rinv, dt, torder=2)
            U,  S  = M_ForwardMap(        U1, S1, A,             dt/2)
        X2 = U @ S @ U.T
        #
        print(f'\tnsteps = {nsteps:5d}\t', end='\t  ')
        print(f'{np.linalg.norm(X1-XRK)/n:8.2e} (1)', end='\t\t  ')
        print(f'{np.linalg.norm(X2-XRK)/n:8.2e} (2)', end='\t')
        print(f'{Tend:4.1f}   {dt:8.6f}')
        