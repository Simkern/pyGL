#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:40:10 2023

@author: skern
"""
import numpy as np

def enorm(a):
    
    return np.real(np.dot(a.conj(),a))