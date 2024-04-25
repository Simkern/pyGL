#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:12:43 2023

@author: skern
"""
import scipy as sp

## Runge-Kutta 4

def rk4_abc(k):

    icase = k % 4
    if icase == 0:
        a  = 8/17
        b  = 0
        c  = 0
        #dtc = dt*(8/17);
    elif icase == 1:
        a  = 17/60
        b  = -15/68
        c  = 8/17
        #dtc = dt*(8/15);
    elif icase == 2:
        a  = 5/12
        b  = -17/60
        c  = 8/15
    else:   # icase == 3
        a  = 3/4
        b  = -5/12
        c  = 2/3
        #dtc = dt*2/3;

    return a,b,c

def RK4_L_advance(q, L, fa, fb, dt):
    
    I   = sp.eye(L.shape[0])

    for k in range(4):
        a,b,__ = rk4_abc(k)
    
        cf = a+b
        a   =  I - 0.5*cf*dt*L
        b   = (I + 0.5*cf*dt*L) @ q + a*fa + b*fb
        q   = sp.linalg.spsolve(a,b)
    
    return q