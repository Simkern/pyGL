import numpy as np

def enormab(a,b):
    """
    Compute the (energy) inner product of two (complex) state vectors a and b
    
    <a,b> = b^H a
    
    """
    return np.dot(b.conj().T,a)


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

def p(x, name=' ', f='E'):
    try:
        d2 = x.shape[1]
        pmat(x, name, f)
    except:
        pvec(x, name)

def pmat(a, name=' ', f='E'):
    print(f'\npmat: {name}')
    for row in a:
        for col in row:
            if f=='E':
                print("{:10.3e}".format(col), end=" ")
            else:
                print("{:8.5f}".format(col), end=" ")
        print("")

def pvec(v, name=' '):
    print(f'\npvec: {name}')
    l = v.size
    lp = 2
    lmax = 2*lp
    if l > lmax:
        a = v[:lp]
        b = v[-lp:]
        split = True
    else:
        a = v
        split = False
    
    if any(np.iscomplex(a)):
        for i, data in enumerate(a):
            dre = np.real(data)
            dim = np.imag(data)
            print(f'\t{i+1:3d}: {dre:8.3f} + {dim:8.3f} i')
        if split:
            print('            ...')
            for i, data in enumerate(b):
                dre = np.real(data)
                dim = np.imag(data)
                print(f'\t{l-lp+i+1:3d}: {dre:8.3f} + {dim:8.3f} i')
    else:
        for i, data in enumerate(a):
            print(f'\t{i+1:3d}: {np.real(data):8.3f}')
        if split: 
            print('            ...')
            for i, data in enumerate(b):
                dre = np.real(data)
                dim = np.imag(data)
                print(f'\t{l-lp+i+1:3d}: {dre:8.3f} + {dim:8.3f} i')
