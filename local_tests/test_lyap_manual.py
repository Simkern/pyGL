import numpy as np
import time

from scipy import linalg, sparse
import sys
import matplotlib.pyplot as plt

import lyap

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

nrmx0 = linalg.norm(Xref, 'fro')

print(np.allclose(A @ Xref + Xref @ A.T, -B @ B.T))

D = linalg.eigvals(A)

a0 = c0 = -max(np.real(D))
b0 = d0 = -min(np.real(D))

popt,__ = lyap.get_param(a0, b0, c0, d0, 2)
# real shifts
#pin = -popt
# cc shifts
pin = np.array( [ -popt[0]+0.1*1j, -popt[0]-0.1*1j ] )
pin = np.append(pin, -popt[1:])
print(pin)

#popt,__ = lyap.get_param(a0, b0, c0, d0, 2)
#Z, ires, res, res_rel, nrmx, nrmz, nrmz_rel = lyap.lrcfadi(A, B, -popt, 100,'niter', Xref)

p_v = np.array([])

l = pin.size

eps = 1e-12
for i in range(l):
    if np.abs(np.imag(pin[i])) < eps:
        pin[i] = np.real(pin[i])

for (i,p) in enumerate(pin[:-1]):
    if not np.abs(p - pin[i+1]) < eps and not p.conj() == pin[i+1]:
        p_v = np.append(p_v,p)
        
if np.imag(pin[-1]) == 0:
    p_v = np.append(p_v,pin[-1])
    
l = p_v.size
    
I = np.eye(A.shape[0])

lu_rl = []
lu_cc = []
is_real = np.imag(p_v) == 0

for i, (shift, is_r) in enumerate(zip(p_v,is_real)):
    if is_r:
        lu, piv = linalg.lu_factor(A + shift*I)
        lu_rl.append((lu,piv))
        lu_cc.append((0,0))
    else:
        s_i = 2*np.real(shift)
        t_i = np.abs(shift)**2
        lu, piv = linalg.lu_factor(A @ A + s_i*A + t_i*I)
        lu_rl.append((0,0))
        lu_cc.append((lu,piv))
         
# for i in range(l):
#     print(f'p[{i}] = {p_v[i]}')
#     if is_real[i]:
#         if not lu_cc[i][0] == 0:
#             print('error')
#         else:
#             print('  real: ok!')
#     else:
#         if not lu_rl[i][0] == 0:
#             print('error')
#         else:
#             print('  complex: ok!')
          
#print(p_v)
#p_v = np.flip(p_v)
#lu_rl.reverse()
#lu_cc.reverse()
p_v = p_v[:1]   # pick first n p[:n]
print(p_v)
l = p_v.size

print(linalg.norm(Xref, 'fro')/nrmx0)
## Start loop
p_old = 1

i  = 0
ip = 0

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

q = np.sqrt(-2*np.real(p))
if is_real:
    print(f'i = {i}: p     = {np.real(p):.2f} :                     real')
    #lu, piv = linalg.lu_factor(A + p*I)
    #V = q*linalg.lu_solve((lu, piv), B)
    V1t = linalg.lu_solve(lu_rl[ip], B)
    #Vb = q*linalg.lu_solve(lu_rl[ip], B)
    Vb = q*V1t
    V2t = 0*V1t # dummy
else:
    print(f'i = {i}: p     = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    q1 = 2*np.sqrt(-np.real(p))*np.abs(p)
    q2 = 2*np.sqrt(-np.real(p))
    
    #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
    #V1 = q1*linalg.lu_solve((lu,piv), B)
    #V1 = q1*linalg.lu_solve(lu_cc[ip], B)
    #V2 = q2*(A @ V1)
    #Vb  = np.column_stack([V1, V2])
    V1t = linalg.lu_solve(lu_cc[ip], B)
    V1  = q1*V1t
    V2t = A @ V1t
    V2  = q2*V2t
    Vb  = np.column_stack([ V1, V2 ])
Z = Vb
    
p_old   = p
V1t_old = V1t
V2t_old = V2t
i = i + 1
ip = i % l       # update shift index

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

if is_real:
    print(    f'i = {i}: p     = {np.real(p):.2f} :                     real')
    q  = np.sqrt(-2*p)
    if is_real_old:
        print(f'       p_old = {np.real(p_old):.2f} :                     real')
        q1 = (p + p_old)
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V - (p + p_old)*linalg.lu_solve((lu,piv),V)
        #Vnew = Vb - (p + p_old)*linalg.lu_solve(lu_rl[ip],Vb)
        V1t =   V1t_old \
              - q1*linalg.lu_solve(lu_rl[ip],V1t_old)
    else: # p_(i-1) complex
        print(f'         p_old = {np.real(p_old):.2f} + i {np.imag(p_old):.2f}: complex conjugate')
        q1 = 2*np.real(p_old) + p
        q2 = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve((lu,piv),V1)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve(lu_rl[ip],V1)
        V1t =      V1t_old \
              - q1*V2t_old \
              + q2*linalg.lu_solve(lu_rl[ip],V2t_old)
    Vnew = q*V1t
    V2t  = 0*V1t
else:  # p_i complex
    print(f'i = {i}: p = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    if is_real_old:
        Vtmp =     A @ V1t_old \
               - p_old*V1t_old
        V1t  = linalg.lu_solve(lu_cc[ip],Vtmp)
        #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        #Vnew = linalg.lu_solve((lu,piv),Vt)
        #Vnew = linalg.lu_solve(lu_cc[ip],Vt)
    else: # p_(i-1) complex
        q1   = np.abs(p_old)**2 - np.abs(p)**2
        q2   = 2*np.real(p_old + p)
        Vtmp =   q1*V1t_old \
               - q2*V2t_old        
        #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        #Vnew = V1 + linalg.lu_solve((lu,piv), Vt)
        #Vnew = V1 + linalg.lu_solve(lu_cc[ip], Vt)
        V1t  =   V1t_old \
               + linalg.lu_solve(lu_cc[ip],Vtmp)
    V2t  = A @ V1t
    V1   = 2*np.sqrt(-np.real(p))*np.abs(p)*V1t
    V2   = 2*np.sqrt(-np.real(p))*V2t
    Vnew = np.column_stack([ V1, V2 ])  
# add new column(s)
Z = np.column_stack([ Z, Vnew ])

p_old   = p
V1t_old = V1t
V2t_old = V2t
i = i + 1
ip = i % l       # update shift index   

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)

p = p_v[ip]
is_real = np.imag(p) == 0
is_real_old = np.imag(p_old) == 0

if is_real:
    print(    f'i = {i}: p     = {np.real(p):.2f} :                     real')
    q  = np.sqrt(-2*p)
    if is_real_old:
        print(f'       p_old = {np.real(p_old):.2f} :                     real')
        q1 = (p + p_old)
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V - (p + p_old)*linalg.lu_solve((lu,piv),V)
        #Vnew = Vb - (p + p_old)*linalg.lu_solve(lu_rl[ip],Vb)
        V1t =   V1t_old \
              - q1*linalg.lu_solve(lu_rl[ip],V1t_old)
    else: # p_(i-1) complex
        print(f'         p_old = {np.real(p_old):.2f} + i {np.imag(p_old):.2f}: complex conjugate')
        q1 = 2*np.real(p_old) + p
        q2 = np.abs(p_old)**2 + 2*p*np.real(p_old) + p**2
        #lu, piv = linalg.lu_factor(A + p*I)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve((lu,piv),V1)
        #Vnew = V2 - q1*V1 + q2*linalg.lu_solve(lu_rl[ip],V1)
        V1t =      V1t_old \
              - q1*V2t_old \
              + q2*linalg.lu_solve(lu_rl[ip],V2t_old)
    Vnew = q*V1t
    V2t  = 0*V1t
else:  # p_i complex
    print(f'i = {i}: p = {np.real(p):.2f} + i {np.imag(p):.2f}: complex conjugate')
    if is_real_old:
        Vtmp =     A @ V1t_old \
               - p_old*V1t_old
        V1t  = linalg.lu_solve(lu_cc[ip],Vtmp)
        #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        #Vnew = linalg.lu_solve((lu,piv),Vt)
        #Vnew = linalg.lu_solve(lu_cc[ip],Vt)
    else: # p_(i-1) complex
        q1   = np.abs(p_old)**2 - np.abs(p)**2
        q2   = 2*np.real(p_old + p)
        Vtmp =   q1*V1t_old \
               - q2*V2t_old        
        #lu, piv = linalg.lu_factor(A @ A + 2*np.real(p)*A + np.abs(p)**2*I)
        #Vnew = V1 + linalg.lu_solve((lu,piv), Vt)
        #Vnew = V1 + linalg.lu_solve(lu_cc[ip], Vt)
        V1t  =   V1t_old \
               + linalg.lu_solve(lu_cc[ip],Vtmp)
    V2t  = A @ V1t
    V1   = 2*np.sqrt(-np.real(p))*np.abs(p)*V1t
    V2   = 2*np.sqrt(-np.real(p))*V2t
    Vnew = np.column_stack([ V1, V2 ])  
# add new column(s)
Z = np.column_stack([ Z, Vnew ])

print(linalg.norm(Xref - Z @ Z.T, 'fro')/nrmx0)