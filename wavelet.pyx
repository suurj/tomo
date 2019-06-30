import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sp
from cython.parallel import prange
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import math
from libc.math cimport sqrt,fabs,exp, cos, tan,sin,M_SQRT2,M_PI,abs
from libcpp.list cimport list as cpplist
import time

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
cdef inline int modulo (int x, int y) nogil:
    return (x % y + y) %y

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def waveletonce(g,h,n):
    cdef int nsig = n 
    cdef int nhalf = n/2
    g = np.ravel(g)
    h = np.ravel(h)
    cdef int fl = g.shape[0]  
    cdef int fl2 = fl/2
    if ((np.mod(n,2) != 0) or (np.mod(h.shape[0],2) != 0 ) or (np.mod(g.shape[0],2) != 0 ) or (g.shape[0] != h.shape[0])):
        raise Exception('Signal or filter length is not divisible by 2.')
        
    fg = np.flip(g,0)
    fh = np.flip(h,0)
    cdef double [:] fgv  = fg
    cdef double [:] fhv  = fh
    cdef int row, col, i,c
    cdef cpplist[int] Wrow
    cdef cpplist[int] Wcol
    cdef cpplist[double] Gdata
    cdef cpplist[double] Hdata
    with nogil:
        for row in range(0,nhalf):
            for col in range (0,fl):
                c = modulo(-fl2 + col +1 + (row) * 2,nsig)
                #c = ((-fl2 + col +1 + (row) * 2) % nsig)
                #cc = ((-fl2 + col +1 + (row) * 2) % n)     
                #print(c,cc)
                Wrow.push_back(row)
                Wcol.push_back(c)
                Hdata.push_back(fgv[col])
                Gdata.push_back(fhv[col])
    
    
    cdef int Nel = Wrow.size()
    coo_row  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_rowv  = coo_row
    coo_col  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_colv  = coo_col
    Hcoo_data = np.zeros((Nel,))
    Gcoo_data = np.zeros((Nel,))
    cdef double [:] Hcoo_datav  = Hcoo_data
    cdef double [:] Gcoo_datav  = Gcoo_data

    for i in range(Nel):
        coo_rowv[i] = Wrow.front()
        Wrow.pop_front()
        
        coo_colv[i] = Wcol.front()
        Wcol.pop_front()

        Hcoo_datav[i] = Hdata.front()
        Hdata.pop_front()
        Gcoo_datav[i] = Gdata.front()
        Gdata.pop_front()
    
    
    GG=coo_matrix((Gcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #GG = csc_matrix(GG)
    HH=coo_matrix((Hcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #HH = csc_matrix(HH)
    P = sp.vstack([HH,GG],format='csc')
    
    return P        