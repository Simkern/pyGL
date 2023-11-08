import numpy as np
import scipy
import sys

import matplotlib.pyplot as plt

sys.path.append('..')

from solvers.arnoldi import arn, arn_inv

plt.close('all')

# generate random matrix
n = 400
rho= np.sqrt(np.random.rand(n))
th = np.random.rand(n)*2*np.pi
d = rho*np.exp(1j*th)

V = np.random.rand(n,n)

A = np.linalg.inv(V) @ np.diag(d) @ V

D,X = scipy.linalg.eig(A)

b = np.ones((n,))

# Compute the set of Ritz values R+ for A
ka = 50
Qa, Ha = arn(A,b,ka)
Da,Va  = scipy.linalg.eig(Ha[0:ka,0:ka])

# Compute the set of Ritz values R- for A^-1
kb = 20
Qb, Hb = arn_inv(A,b,kb)
Dbtmp,Vbtmp = scipy.linalg.eig(Hb[0:kb,0:kb])
Db = [ 1/r for r in Dbtmp ]

fig = plt.figure(2)
ax = plt.plot(np.real(Da),np.imag(Da),'ro', mfc='none',  label='Ritz values R+')
plt.plot(np.real(Db),np.imag(Db),'bo', mfc='none',  label='Ritz values 1/R-')
plt.plot(np.real(D),np.imag(D),'k+', label = 'Eigenvalues A')
plt.axis('square')
plt.legend()
plt.show()