#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:18:43 2024

@author: skern
"""
import sys
import copy as cp
import numpy as np
from scipy.linalg import qr
from git.core.utils import p

n = 6
m = 4
#kdim = 6

#A = np.random.rand(n,m)

A = np.array([[1.0,2.0,3.0,4.0],[2.0,6.0,3.0,1.0],
              [5.0,4.0,3.0,2.0],[1.0,1.0,1.0,1.0],
              [2.0,2.0,2.0,2.0],[0.0,0.0,1.0,1.0]])

A[:,3] = A[:,0] + 1e-10*np.random.rand(n,)
#A[:,5] = A[:,1] + 1e-10*np.random.rand(8,)


Q,R,P = qr(A, pivoting=True)

#p(Q)
#p(R)

#p(A)

Rii = np.zeros((m,))

for i in range(m):
    Rii[i] = np.dot(A[:,i],A[:,i])

R = np.zeros((m,m))
P = np.zeros((m,m))
Qwrk = A.copy()

Q = A.copy()

p(Qwrk,'A')

card = np.arange(m)
print(card)

for j in range(m):
    #p(Rii[j:])
    idx = np.argmax(Rii)
    card[[j,idx]] = card[[idx,j]]
    print(f'j = {j}, idx = {idx}')
    print(card)
    
    #P[idx,j] = 1
    
    #p(Q,f'pre {j} <-> {idx}')
    if not j == m-1: 
        Q[:,[j,idx]] = Q[:,[idx,j]]
        R[:j,[j,idx]] = R[:j,[idx,j]]
    #p(Q,'post')
    #p(P)
 
    alpha = np.linalg.norm(Q[:,j])
    print(alpha)
    R[j,j] = alpha
    Q[:,j] /= alpha
    for i in range(j+1,m):
        proj = Q[:,j].T @ Q[:,i]
        Q[:,i] -= Q[:,j] * proj
        R[j,i] = proj
    Rii[[j,idx]] = Rii[[idx,j]]
    for i in range(m):
        if i <= j:
            Rii[i] = 0.0
        else:
            Rii[i] -= R[j,i]**2
    
for j in range(m):
    P[card[j],j] = 1
    
p(R)
p(Q)
    
p(P)
p(A @ P - Q @ R, 'CHK')
p(R)

sys.exit()

'''
   7.810    3.457    4.993    5.250 
   0.000    3.879    0.963    2.282 
   0.000    0.000    3.023    0.195 
   0.000    0.000    0.000    0.444 
'''


def CGS(X):
    Q = np.zeros_like(X)
    n, r = X.shape
    R = np.zeros((r,r))
    for j in range(n):
        Q[:,j] = X[:,j]
        # project
        proj = Q[:,:j].T @ X[:,j]
        Q[:,j] -= Q[:,:j] @ proj
        R[:j,j] = proj
        # normalise
        alpha = np.linalg.norm(Q[:,j])
        R[j,j] = alpha
        Q[:,j] /= alpha
    return Q, R

def DGS(X):
    Q = np.zeros_like(X)
    n, r = X.shape
    R = np.zeros((r,r))
    for j in range(n):
        Q[:,j] = X[:,j]
        # project
        proj = Q[:,:j].T @ X[:,j]
        Q[:,j] -= Q[:,:j] @ proj
        R[:j,j] = proj
        # project round 2
        proj = Q[:,:j].T @ Q[:,j]
        Q[:,j] -= Q[:,:j] @ proj
        R[:j,j] += proj
        # normalise
        alpha = np.linalg.norm(Q[:,j])
        R[j,j] = alpha
        Q[:,j] /= alpha
    return Q, R

def MGS(X):
    Q = cp.copy(X)
    n, r = X.shape
    R = np.zeros((r,r))
    for j in range(n):
        alpha = np.linalg.norm(Q[:,j])
        R[j,j] = alpha
        Q[:,j] /= alpha
        for i in range(j+1,n):
            proj = Q[:,j].T @ Q[:,i]
            Q[:,i] -= Q[:,j] * proj
            R[j,i] = proj
    return Q, R

n = 10
eps = 1e-6
X = np.random.rand(n,n)

X = np.zeros((n,n))
X[:,0] = np.random.rand(n,)
for i in range(1,n):
    rcol = np.random.rand(n,)
    X[:,i] = X[:,i-1] + eps*rcol 
    
print(np.linalg.det(X))
'''
X = np.diag(np.ones((n,))*eps)
X[:,0] = eps
X[0,:] = 1
'''
Q1,R1 = CGS(X)
print('\nCGS orthonormality: {0:e}'.format(np.linalg.norm(np.eye(n) - Q1.T @ Q1)))
print('CGS decomposition:  {0:e}'.format(np.linalg.norm(X - Q1 @ R1)))
Q2,R2 = DGS(X)
print('\nDGS orthonormality: {0:e}'.format(np.linalg.norm(np.eye(n) - Q2.T @ Q2)))
print('DGS decomposition:  {0:e}'.format(np.linalg.norm(X - Q2 @ R2)))
Q3,R3 = MGS(X)
print('\nMGS orthonormality: {0:e}'.format((np.linalg.norm(np.eye(n) - Q3.T @ Q3))))
print('MGS decomposition:  {0:e}'.format(np.linalg.norm(X - Q3 @ R3)))




