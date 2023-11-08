# pyGL

Python toolbox for the solution of the complex Ginzburg-Landau equations using finite differences

Summary:
1. Differentiation matrices for the second-order finite difference scheme
2. Integration routines for the linear and non-linear CGL equations based on the Crank-Nicolson discretisation of the PDEs.
3. Testing suites for the computation of the eigenvalue spectrum and the optimal initial condition for maximum transient growth via matrix exponentiation and direct-adjoint looping.

Matrix-free tools for solution of Lyapunov equations

Summary:
1. Implementations of the Low-rank Cholesky-Factor ADI method (LRCFADI) for real matrices (involving only real algebra) for the case of purely real and complex conjugate shifts
2. Utils including Gram-Schmidt and Arnoldi factorisations to compute suboptimal shift heuristics
3. Optimal shift calculation for the case of symmetric system matrices A
