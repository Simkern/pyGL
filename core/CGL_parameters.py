import numpy as np

def CGL(mu0,cu = 0.2, cd = -1, U = 2, verb = False):

    mu    = mu0 - cu**2
    nu    = U + 1j*cu
    gamma = 1.0 + 1j*cd

    Umax  = U + 2*cu*cd
    mu_t = Umax**2/(4*abs(gamma)**2) # - 2*cu**2

    if verb:
        print(f'Complex Ginzburg-Landau equations for parallel flows:\n')
        print(f'  U     = {U:.2f}')
        print(f'  cu    = {cu:.2f}')
        print(f'  cd    = {cd:.2f}')
        print(f'  mu_0  = {mu0:.2f}\n')
        print(f'  mu    = {mu:.2f}')
        print(f'  nu    = {nu:.2f}')
        print(f'  gamma = {gamma:.2f}\n')
        print(f'  Group velocity:\n    Umax  = {Umax:.2f}')
        print(f'  Threshold local absolute instability:  mu_0 > mu_t\n    mu_t = {mu_t:.6f}\n')

    return mu,nu,gamma,Umax,mu_t

def CGL2(x, mu0, cu = 0.1, cd = -1, U = 2, mu2 = -0.01, verb = False):

    mu_scal,nu,gamma,Umax,mu_t = CGL(mu0,cu,cd,U)

    mu = mu_scal*np.ones(x.shape) + mu2*x**2/2

    h = np.sqrt(-2*mu2*gamma)
    mu_c = mu_t + np.abs(h)/2*np.cos(np.angle(gamma)/2)

    if verb:
        print(f'Complex Ginzburg-Landau equations for spatially developing flows:\n')
        print(f'  U     = {U:.2f}')
        print(f'  cu    = {cu:.2f}')
        print(f'  cd    = {cd:.2f}')
        print(f'  mu_0  = {mu0:.2f}\n')
        print(f'  mu    = {mu_scal:.2f} + {mu2:.2f}*x**2/2')
        print(f'  nu    = {nu:.2f}')
        print(f'  gamma = {gamma:.2f}\n')
        print(f'  Group velocity:\n    Umax  = {Umax:.2f}')
        print(f'  Threshold local absolute instability:  mu_0 > mu_t')
        print(f'    mu_t = {mu_t:.6f}')
        print(f'  Threshold global instability:          mu_0 > mu_c')
        print(f'    mu_c = {mu_c:.6f}\n')
        if np.max(mu) > 0:
            x12 = np.sqrt(-2*mu_scal/mu2)
            print(f'  Region of instability:')
            print(f'    xI,xII = +/- {x12:.2f}\n')

    return mu,nu,gamma,Umax,mu_t