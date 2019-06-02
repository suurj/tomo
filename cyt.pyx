import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sp
from cython.parallel import prange
from scipy.linalg import circulant
from scipy.sparse import csr_matrix,csc_matrix
from libc.math cimport sqrt,fabs,exp


def csr_spmul(int Nrow,int Ncol,np.ndarray[np.double_t] data, np.ndarray[int] indices,np.ndarray[int] ptr, np.ndarray[np.double_t, ndim=2] x):
    cdef int i
    cdef int j
    Nrow = ptr.shape[0]-1
    cdef np.ndarray r  = np.zeros((Nrow,1))
    for i in range(0,Nrow):
        for j in range(ptr[i],ptr[i+1]):
            r[i] += data[j] * x[indices[j]]

    return r
@cython.boundscheck(False) 
@cython.wraparound(False) 
def csc_spmul(int Nrow,int Ncol,np.ndarray[np.double_t] data, np.ndarray[int] indices, np.ndarray[int] ptr, np.ndarray[np.double_t, ndim=2] x):
    r  = np.zeros((Nrow,1))
    cdef double[:, :] rv = r
    cdef double[:] dv = data
    cdef int[:] ptrv = ptr
    cdef int[:] iv = indices
    cdef double[:, :] xv = x
    cdef int i
    cdef int j
    for i in prange(Ncol,nogil=True):
        for j in range(ptrv[i],ptrv[i+1]):
            rv[iv[j],0] += dv[j] * xv[i,0]

    return r

@cython.boundscheck(False)
@cython.wraparound(False)   
def csc_col(int Nrow,int Ncol,np.ndarray[np.double_t] data, np.ndarray[int] indices, np.ndarray[int] ptr, int col):
    r  = np.zeros((Nrow,1))
    cdef double[:, :] rv = r
    cdef int[:] ptrv = ptr
    cdef double[:] dv = data
    cdef int j
    cdef int start = ptrv[col]
    cdef int stop = ptrv[col+1]
    for j in prange(start,stop,1,nogil=True):
        rv[indices[j],0] = dv[j]

    return r

@cython.boundscheck(False) 
@cython.wraparound(False) 
cdef csc_vmul(int Nrow,int Ncol,double[:] data, int[:] indices, int[:] ptr, double[:,:] x):
    r  = np.zeros([Nrow, 1], dtype=np.double)
    cdef double[:, :] rv = r
    cdef int i
    cdef int j
    for i in prange(Ncol,nogil=True):
        for j in range(ptr[i],ptr[i+1]):
            rv[indices[j],0] += data[j] * x[i,0]

    return r

'''
cdef lhstep(double[:] Mdata, int[:] Mindices, int[:] Mptr,double [:,:] y,double[:,:] previous,double pnorm,double newcomp, int col):
    cdef int j
    cdef int start = ptr[col]
    cdef int stop = ptr[col+1]
    cdef double change = 0
    for j in prange(start,stop,1,nogil=True):
        change += (newcomp*data[j] - y[indices[j],0])**2.0 - previous[indices[j],0]
'''    
    

@cython.boundscheck(False)
@cython.wraparound(False) 
def mwg(M, Lx, Ly,  y, x0,N, regalpha=1, samplebeta=0.3, sampsigma=1,lhsigma=1):
    dim = y.shape[0]
    x = x0
    xc = np.copy(x)
    
    if not isinstance(M, sp.csc.csc_matrix):
        M = csc_matrix(M)
    
    if not isinstance(Lx, sp.csc.csc_matrix):
        Lx = csc_matrix(Lx)
        
    if not isinstance(Ly, sp.csc.csc_matrix):
        Ly = csc_matrix(Ly)    

    cdef double alpha = regalpha
    cdef double samplingsigma = sampsigma
    cdef double beta = samplebeta
    cdef double Ci = 1/lhsigma
    
    cdef double[:] Mdata =  M.data
    cdef int[:] Mindices =  M.indices
    cdef int[:] Mptr =  M.indptr
    
    cdef double[:] Lxdata =  Lx.data
    cdef int[:] Lxindices =  Lx.indices
    cdef int[:] Lxptr =  Lx.indptr
    
    cdef double[:] Lydata =  Ly.data
    cdef int[:] Lyindices =  Ly.indices
    cdef int[:] Lyptr =  Ly.indptr
 
    
    
    def pdf(xf):
        #return np.ravel(np.exp(-Ci * (M @ x - y).T @ (M @ x - y)))
        return np.ravel(np.exp( -1/2*Ci*(M @ xf - y).T  @ (M @ xf - y)))*np.ravel(np.exp(-alpha*np.sum(np.abs(Lx @ xf))))*np.ravel(np.exp(-alpha*np.sum(np.abs(Ly@xf))))
    
    def pdff(xf):
        #return np.ravel(np.exp(-Ci * (M @ x - y).T @ (M @ x - y)))
        return (np.sum((M @ xf - y).T  @ (M @ xf - y)) ,np.sum(np.abs(Lx @ xf)) ,np.sum(np.abs(Ly @ xf)) )
    #return np.ravel(np.exp(-1.0 / 2.0 * (M@x-y).T @ Ci @ (M@x-y)))

    
    chain = np.zeros((dim, N))
    chain[:,0] = np.ravel(x)
   
    
    cdef int acc = 0
    w = M@x
    w2 = Lx@x
    w3 = Ly@x
    value = np.sum((w-y).T@(w-y))
    value2 = np.sum(np.abs(w2))
    value3 = np.sum(np.abs(w3))
    
    cdef double[:, :] lhcompv = w
    cdef double[:, :] prcompv = w2
    cdef double[:, :] prcompv2 = w3
    cdef double likelihood = value
    cdef double prior = value2
    cdef double prior2 = value3
    
   
    cdef int i
    cdef int j
    cdef double[:, :] xv = x
    cdef double[:, :] xcv = xc
    cdef double[:, :] yv = y
    cdef double new,change, change2, change3, old
    cdef double[:, :] chainv = chain
  
    cdef int k, start, stop
    cdef double[:] acceptv,number
    #print(y)
    #print(w)
    #print(likelihood)
    for i in range(1,N):
        randoms = np.random.randn(dim,)
        accept = np.random.rand(dim,)
        acceptv = accept
        number = randoms
        
        for j in range(0,dim):
            
            old = chainv[j,i-1]
            new = sqrt(1-beta**2.0)*old + beta*samplingsigma*samplingsigma*number[j]
            #print(new)
            
            change = 0
            change2 = 0
            change3 = 0
            
            start = Mptr[j]
            stop = Mptr[j+1]
            
            for k in prange(start,stop,1,nogil=True):
                change += -(lhcompv[Mindices[k],0]- yv[Mindices[k],0])**2.0 + (Mdata[k]*(new-old) + lhcompv[Mindices[k],0] - yv[Mindices[k],0])**2.0 
                #print(Mdata[k],yv[Mindices[k],0], lhcompv[Mindices[k],0])
                #change += (new*Mdata[k] - yv[Mindices[k],0])**2.0 - lhcompv[Mindices[k],0]
                
            start = Lxptr[j]
            stop = Lxptr[j+1]    
                
            for k in prange(start,stop,1,nogil=True):
                change2 += -fabs(prcompv[Lxindices[k],0] ) + fabs(Lxdata[k]*(new-old) + prcompv[Lxindices[k],0])
                
            start = Lyptr[j]
            stop = Lyptr[j+1]    
                
            for k in prange(start,stop,1,nogil=True):
                change3 += -fabs(prcompv2[Lyindices[k],0] ) + fabs(Lydata[k]*(new-old) + prcompv2[Lyindices[k],0])    
             
            xc[j,0] = new
            #dd = (likelihood+change,prior+change2)
            #ddf = (likelihood,prior)
            #cc = np.exp(-1/2*Ci*(likelihood+change))*np.exp(-alpha*(prior+change2))/(np.exp(-1/2*Ci*(likelihood))*np.exp(-alpha*(prior)))
            ratio = exp(-1/2*Ci*change -alpha*change2 - alpha*change3)
            #print(ratio)
            #print(pdf(xc)/pdf(x))
            
            if(acceptv[j] <= ratio):
                chainv[j,i] = new
                likelihood = likelihood + change
                prior = prior + change2
                prior2 = prior2 + change3
                
                start = Lxptr[j]
                stop = Lxptr[j+1]    
                
                for k in prange(start,stop,1,nogil=True):
                    prcompv[Lxindices[k],0] = prcompv[Lxindices[k],0] + Lxdata[k]*(new-old)
                    
                start = Lyptr[j]
                stop = Lyptr[j+1]    
                
                for k in prange(start,stop,1,nogil=True):
                    prcompv2[Lyindices[k],0] = prcompv2[Lyindices[k],0] + Lydata[k]*(new-old)    
                    
                start = Mptr[j]
                stop = Mptr[j+1]
            
                for k in prange(start,stop,1,nogil=True):
                    lhcompv[Mindices[k],0] = lhcompv[Mindices[k],0] + Mdata[k]*(new-old)
                    
                #x[j,0] = new
                
            else:
                chainv[j,i] = old
                #xc[j,0] = old
            #change = np.reshape(M[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #change2 = np.reshape(L[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #newvalue=np.exp((change-y+w).T@(change-y+w)*Ci*-1/2)
            #newvalue = np.ravel(np.exp((change - y + w).T @ (change - y + w) * -Ci))
            #newvalue2 = np.exp(-alpha*np.sum(np.abs(change2+w2)))
            #ratio2 = newvalue*newvalue2/(value*value2)
            #xc[j,0] = pr
            #ratio = pdf(xc)/pdf(x)
            #if (np.random.rand(1) <= ratio2):
                #x[j,0] = pr#xc[j,0]
                #w = w+ change
                #w2 = w2+change2
                #value = newvalue
                #e = pdf(x)
                #value2 = newvalue2
                #acc = acc +1
            #else:
                #pass
                #xc[j,0] = x[j,0]

        #numer = numer + x
        #chain[:,i] = np.ravel(x)
        
    #print(acc/(N*dim))
    #print(numer/N)
    return  chain

