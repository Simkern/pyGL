import numpy as np
import sys

from scipy import linalg, sparse
import matplotlib.pyplot as plt

sys.path.append('..')

from solvers.arnoldi import arn, arn_inv
from solvers.lyapunov import lrcfadi
from solvers.lyap_utils import get_opt_shifts, check_shifts

plt.close("all")

def pmat(a):
    for row in a:
        for col in row:
            print("{:8.3f}".format(col), end=" ")
        print("")

plt.close('all')

n = 400
m = 1
h = 1/(n+1)

I = np.eye(n)
# test problem
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape = (n,n)).toarray()
A = A/h
# BC
A[0,0] = -1/h

B = np.zeros((n,m))
B[-1] = 1/h
# B[-1,0] = 1/h
# B[-1,1] = 3
# B[-2,1] = -3

# Benchmark
Xref = linalg.solve_continuous_lyapunov(A, -B @ B.T)

#print(np.allclose(A @ Xref + Xref @ A.T, -B @ B.T))

D = linalg.eigvals(A)

a0 = c0 = -max(np.real(D))
b0 = d0 = -min(np.real(D))

# 200 iterations

niter = 200
fig1 = plt.figure(1)
for power in range(4):
    print(f'n = {power}, popt')
    l = 2**power  # select number of shifts to use
    popt,__ = get_opt_shifts(a0, b0, c0, d0, power)
    Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = lrcfadi(A, B, -popt, niter,'niter', Xref)
    plt.plot(ires,np.log10(nrmx),label=f'l = {l}, popt')
plt.xlabel('iterations')
#plt.ylabel('||X_i - X_ref||_F/ ||X_ref||_F')
plt.ylabel('||X_i - X_ref||_2/ ||X_ref||_2')
plt.legend()
plt.show(block=False)


# Tolerance

fig2 = plt.figure(2)
tol = 1e-8
for power in range(4):
    print(f'n = {power}, popt')
    l = 2**power  # select number of shifts to use
    popt,__ = get_opt_shifts(a0, b0, c0, d0, power)
    Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = lrcfadi(A, B, -popt, tol,'tol', Xref)
    plt.plot(ires,np.log10(nrmx),label='n = 1, popt')
plt.xlabel('iterations')
#plt.ylabel('||X_i - X_ref||_F/ ||X_ref||_F')
plt.ylabel('||X_i - X_ref||_2/ ||X_ref||_2')
plt.legend()
plt.xlim([0,200])
plt.show(block=False)

##

# compute shifts
b0 = np.ones((n,))
# Compute the set of Ritz values R+ for A
ka = 30
__,Ha = arn(A,b0,ka)
pA,__ = linalg.eig(Ha[0:ka,0:ka])
# Compute the set of Ritz values R- for A^-1
kb = 15
__,Hb    = arn_inv(A,b0,kb)
Dbtmp,__ = linalg.eig(Hb[0:kb,0:kb])
pAinv = np.array([ 1/r for r in sorted(Dbtmp) ])

pA    = np.real(sorted(pA))
pAinv = np.real(pAinv)

# check for complex shifts
p_v, status, is_r = check_shifts(pA)
if status >= 0:
    if status == 0:
        is_cmplx = False
    else:
        is_cmplx = True
else:
    sys.exit()

# Sub-optimal shifts (Arnoldi)

fig3 = plt.figure(3)
n = 30
step = 5
for nAinv in range(0,n+step,step):
    nA = n - nAinv
    print(f'n = {n}, pA,pAinv = ({nA},{nAinv})')
    pin = np.random.permutation(np.concatenate((pA[:nA],pAinv[:nAinv])))
    Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = lrcfadi(A, B, pin, tol,'tol', Xref)
    plt.plot(ires,np.log10(nrmz_rel),label=f'n = {n}, pA,pAinv = ({nA},{nAinv})')
plt.legend()
plt.xlabel('iterations')
#plt.ylabel('||X_i - X_ref||_F/ ||X_ref||_F')
#plt.ylabel('||X_i - X_ref||_2/ ||X_ref||_2')
plt.ylabel('\\z_i\\_2/\\Z_i-1\\_2')
plt.show(block=False)
    
    
    
