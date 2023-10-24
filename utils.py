#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def enormab(a,b):
    """
    Compute the (energy) inner product of two (complex) state vectors a and b
    
    <a,b> = b^H a
    
    """
    return np.dot(b.conj(),a)


def enorm(a):
    """
    Compute the (energy) inner product of a (complex) state vector a
    
    <a,a> = a^H a
    
    """
    return np.real(enormab(a,a))

def en(a):
    """
    Compute vector norm from energy inner product
    """
    return np.sqrt(enorm(a))