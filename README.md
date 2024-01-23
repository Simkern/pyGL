# pyGL

Python toolbox for the solution of the complex Ginzburg-Landau equations using finite differences

Summary:
1. Differentiation matrices for the second-order finite difference scheme
2. Integration routines for the linear and non-linear CGL equations based on the Crank-Nicolson discretisation of the PDEs.
3. Testing suites:
   1. Time integration of the CGL equations
   2. Computation of the eigenvalue spectrum of the linear operator
   3. Optimal initial condition for maximum transient growth via matrix exponentiation and direct-adjoint looping
   4. Arnoldi factorization
   5. Algorithms for the solution of the Lyapunov equation AX + XA' = -BB':
        a) Low-rank Cholesky-Factor ADI - LRCFADI (real (LU-based), complex (LU-based), real (GMRES-based))
        b) Krylov plus Inverse-Krylov - K-PIK (real (LU-based), real (GMRES-based))
        c) Low-rank approximation of differential Lyapunov equations using an operator-splitting-based time integrator (real)

Matrix-free tools for solution of Lyapunov equations

Summary:
1. Implementations of the Low-rank Cholesky-Factor ADI method (LRCFADI) for real matrices (involving only real algebra) for the case of purely real and complex conjugate shifts.
   The code for the LRCFADI is based on the matlab implementation of from LYAPACK by Pentzl with adaptations for purely real algebra.
3. Utils including Gram-Schmidt and Arnoldi factorisations to compute suboptimal shift heuristics
4. Optimal shift calculation for the case of symmetric system matrices A
5. Extensions of the solution algorithm to support iterative solutions of the linear systems (GMRES) with or without Laplace-preconditioning.
