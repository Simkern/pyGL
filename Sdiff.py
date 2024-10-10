#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:29:37 2024

@author: skern
"""
import numpy as np
from numpy.linalg import qr, svd, norm

m = 4
n = 10

U, _ = qr(np.random.random_sample((n,m)))
V, _ = qr(np.random.random_sample((n,m)))

S = np.random.random_sample((m,m))
S = 0.5*(S + S.T)
X = np.random.random_sample((m,m))
X = 0.5*(X + X.T)

A = U @ S @ U.T
B = V @ X @ V.T

D = A - B

print("|| A - B ||_2 = ", norm(D))

u,s,v = svd(D)

print(" s(1)    = ", s[0])
print(" || . ||_F  = ", np.sqrt(sum(s**2)))

Vt1 = V.T @ U
Vp, R = qr(V - U @ U.T @ V)

Vt2 = V.T @ Vp

D_LR = np.block([ [ S - Vt1.T @ X @ Vt1, - Vt1.T @ X @ Vt2 ], 
                  [   - Vt2.T @ X @ Vt1, - Vt2.T @ X @ Vt2 ] ])

u,s,v = svd(D_LR)

print(" s_LR(1) = ", s[0])
print(" || . ||_F  = ", np.sqrt(sum(s**2)))
