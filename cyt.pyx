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

class argumentspack():
    __slots__ = ['M', 'Lx', 'Ly', 'y', 's2', 'a' ,'b', 'logdensity', 'gradi']
    def __init__(self,logdensity= lambda x,Q: 1,gradi = lambda x,Q: 0,M=None, Lx=None, Ly=None, y=0, s2=1.0, a=1.0, b=0.01):
        self.M = M
        self.Lx = Lx
        self.Ly = Ly
        self.y = y
        self.s2 = s2
        self.a = a
        self.b = b
        self.logdensity = logdensity
        self.gradi = gradi
        
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)        
def tfun_cauchy( x,Q):
    x = np.reshape(x, (-1, 1))
    Mxy = Q.M.dot(x) - Q.y
    Lxx = Q.Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Q.Ly.dot(x)
    alpha = Q.a
    return   -0.5/Q.s2*Mxy.T.dot( Mxy) - np.sum(np.log(alpha + np.multiply(Lxx,Lxx))) - np.sum(
        np.log(alpha + np.multiply(Lyx,Lyx))) 
    
    #return   -0.5/Q.s2*np.dot(Mxy.T, Mxy) - np.sum(np.log(alpha + np.multiply(Lxx,Lxx))) - np.sum(
    #    np.log(alpha + np.multiply(Lyx,Lyx)))         


@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def tfun_tv( x,Q):
    x = np.reshape(x, (-1, 1))
    Mxy = Q.M.dot(x) - Q.y
    Lxx = Q.Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Q.Ly.dot(x)
    alpha = Q.a
    beta = Q.b
    q =  -0.5/Q.s2*Mxy.T.dot( Mxy) - alpha * np.sum(np.sqrt(np.multiply(Lxx,Lxx) + beta)) - alpha * np.sum(
        np.sqrt(np.multiply(Lyx,Lyx) + beta))
    return (np.ravel(q))  


@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False) 
def tfun_tikhonov(x,Q):
    x = np.reshape(x, (-1, 1))
    Mxy = Q.M.dot(x) - Q.y
    Lxx = Q.Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Q.Ly.dot(x)
    alpha = Q.a
    return -0.5/Q.s2*Mxy.T.dot( Mxy) - alpha*np.dot(Lxx.T,Lxx) - alpha*np.dot(Lyx.T,Lyx)
    #return np.sum(np.array([a,b1,b2]))
      

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def tikhonov_grad(x,Q):
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    a = Q.a
    s2 = Q.s2
    y = Q.y
    Mxy = M.dot(x) - y
    Lxx = Lx.dot(x) 
    Lyx = Ly.dot(x)
    return -1.0/s2  * (M.T).dot(Mxy)  - 2.0*a*(Lx.T).dot(Lxx) - 2.0*a*(Ly.T).dot(Lyx)

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def tv_grad(x,Q):
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    a = Q.a
    s2 = Q.s2
    y = Q.y
    b = Q.b
    cdef double [:] Lxdata =  Lx.data
    cdef int [:] Lxindices = Lx.indices
    cdef int [:] Lxptr = Lx.indptr
    #print(M)
    cdef double [:] Lydata = Ly.data
    cdef int [:]Lyindices = Ly.indices
    cdef int [:]Lyptr = Ly.indptr
    (ro,co) = Ly.shape
    cdef int row = ro
    cdef int col = co
    cdef double alfa = a
    cdef double beta = b 
    

    # Mxy = np.dot(M,x)-y
    Mxy = M.dot(x) - y
    # Lxx = np.dot(Lx,x)
    cdef double [:,:] Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    cdef double [:,:] Lyx = Ly.dot(x)

    #common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -1.0/s2  * (M.T).dot(Mxy) 
    cdef double [:,:] grv = gr
    cdef int i,j
    cdef double s

    #exit(1)
    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - Lxdata[j]*Lxx[Lxindices[j],0]/sqrt(Lxx[Lxindices[j],0]*Lxx[Lxindices[j],0] + beta)

        #exit(0)
        grv[i,0] = grv[i,0] + alfa*s

    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - Lydata[j] * Lyx[Lyindices[j],0] / sqrt(Lyx[Lyindices[j],0] *Lyx[Lyindices[j],0] + beta)

        # exit(0)
        grv[i, 0] = grv[i, 0] + alfa * s 
    return gr

@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)
def cauchy_grad(x,Q):
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    a = Q.a
    s2 = Q.s2
    y = Q.y
    cdef double [:] Lxdata =  Lx.data
    cdef int [:] Lxindices = Lx.indices
    cdef int [:] Lxptr = Lx.indptr
    #print(M)
    cdef double [:] Lydata = Ly.data
    cdef int [:]Lyindices = Ly.indices
    cdef int [:]Lyptr = Ly.indptr
    (ro,co) = Ly.shape
    cdef int row = ro
    cdef int col = co
    cdef double alfa = a

    Mxy = M.dot(x) - y
    #
    cdef double [:,:] Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    cdef double [:,:] Lyx = Ly.dot(x)

    #common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -1.0/s2  * (M.T).dot(Mxy)
    cdef double [:,:] grv = gr
    cdef int i,j
    cdef double s


    #exit(1)
    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - 2*Lxx[Lxindices[j],0]/(alfa+Lxx[Lxindices[j],0]*Lxx[Lxindices[j],0])*Lxdata[j]

        #exit(0)
        grv[i,0] = grv[i,0] + s

    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - 2 * Lyx[Lyindices[j],0] / (alfa + Lyx[Lyindices[j],0] *Lyx[Lyindices[j],0])*Lydata[j]

        # exit(0)
        grv[i, 0] = grv[i, 0] +   s
    return gr

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def leapfrog(theta,r,epsilon,Q,currgrad):
    r = r + epsilon*0.5*currgrad
    theta = theta + epsilon*r
    currdensity = Q.logdensity(theta,Q)
    currgrad = Q.gradi(theta, Q)
    r = r + epsilon*0.5*currgrad
    #print(theta)
    return (theta,r,currdensity,currgrad)


def initialeps(theta,Q,currdensity,currgrad):
    def totalenergy( r, cd):
        # return np.exp(-1/2*x.T@Q@x)
        # return np.exp(-1/2*np.sum(np.power(R@x,2)))
        energy = (cd - 0.5 * np.dot(r.T, r))
        # if (x[0] > 10):
        #    print(x)
        return energy
    eps = 1.0
    r = np.random.randn(theta.shape[0],1)
    (theta2,r2,currdensity2,currgrad2) = leapfrog(theta,r,eps,Q,currgrad)
    a = 2.0*(np.exp(totalenergy(r2,currdensity2) - totalenergy(r,currdensity)) > 0.5) -1.0
    #print(totalenergy(theta2,r2,Q) - totalenergy(theta,r,Q),a)

    while  a * (totalenergy(r2,currdensity2) - totalenergy(r,currdensity)) > -a * np.log(2):
        eps = 2.0**(a)*eps
        (theta2,r2,currdensity2,currgrad2) = leapfrog(theta,r,eps,Q,currgrad)
    print(eps)
    return eps

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)  
def buildtree(theta,r,u,v,j,epsilon,theta0,r0,Q,initialdr,currgrad):
    if (j==0):
        thetatilde, rtilde, currdensitytilde,currgradtilde = leapfrog(theta,r,v*epsilon,Q,currgrad)
        #ldensitytilde = Q.logdensity(thetatilde,Q)
        #ldensity = Q.logdensity(theta0,Q)
        logu = np.log(u)
        #print(currdensity)
        #print(initialdr)
        diff = np.min(np.array([np.exp(currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) -initialdr),1]))
        ntilde = int(logu <= (currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        stilde = int(logu < (1000.0+currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        #print(diff)
        return thetatilde,rtilde,thetatilde,rtilde,thetatilde,ntilde,stilde,diff,1,currdensitytilde,currgradtilde,currgradtilde,currgradtilde

    else:
        thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde,currdensitytilde,currgradtilde,gradplus,gradminus = buildtree(theta,r,u,v,j-1,epsilon,theta0,r0,Q,initialdr,currgrad)

        if(stilde == 1):
            if(v == -1):
                thetaminus, rminus, _, _, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde,currdensitytildetilde,currgradtildetilde,_,gradminus = buildtree(
                    thetaminus, rminus, u, v, j - 1, epsilon, theta0, r0,Q,initialdr,gradminus)
            else:
                _, _, thetaplus, rplus, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde,currdensitytildetilde,currgradtildetilde,gradplus,_ = buildtree(
                    thetaplus, rplus, u, v, j - 1, epsilon, theta0, r0,Q,initialdr,gradplus)
            if(np.random.rand() <= ntildetilde/np.max(np.array([ntilde + ntildetilde,1]))):
                thetatilde = thetatildetilde
                currdensitytilde = currdensitytildetilde
                currgradtilde = currgradtildetilde

            alfatilde = alfatilde + alfatildetilde
            nalfatilde = nalfatilde + nalfatildetilde
            stilde = stildetilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus)>=0)
            ntilde = ntilde + ntildetilde
            #print(";;", stilde)
        return thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde,currdensitytilde,currgradtilde,gradplus,gradminus


#@profile
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def hmc(M,theta0,Q,Madapt,de=0.6,cm=False):
    #np.random.seed(1)
    theta0 = np.reshape(theta0,(-1,1))
    dim = theta0.shape[0]
    if (cm == False):
        theta = np.zeros((dim, M))
        theta[:, 0] = np.ravel(theta0)
    delta = de
    currdensity = Q.logdensity(theta0, Q)
    currgrad = Q.gradi(theta0, Q)
    epsilon = initialeps(theta0,Q,currdensity,currgrad)
    myy = np.log(10*epsilon)
    epsilonhat = 1.0
    Hhat = 0.0
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    cmestimate = np.zeros((dim,1))
    if (Madapt >= M):
        raise Exception('Madapt <= M.')
    if (de < 0.05 or de > 0.95):
        raise Exception('Delta is not within reasonable range.')

    for i in range(1,M):
        print(i)
        r0 = np.random.randn(dim, 1)
        u = np.exp(currdensity - 0.5*np.dot(r0.T,r0))*np.random.rand()
        initialdr = currdensity - 0.5 * np.dot(r0.T, r0)
        thetaminus = np.reshape(theta0,(-1,1))
        thetaplus = np.reshape(theta0,(-1,1))
        rminus = r0
        rplus = r0
        gradminus = currgrad
        gradplus = currgrad
        j = 0
        n = 1.0
        s = 1.0
        if(cm == False):
            theta[:, i] = theta[:, i - 1]
        #exit(1)
        #print(np.ravel(r0))
        while (s==1):
            vj = np.random.choice(np.array([-1,1]))
            #vj = int(2 * (np.random.rand() < 0.5) - 1)
            if (vj ==-1):
                thetaminus,rminus,_,_,thetatilde,ntilde,stilde,alfa,nalfa,currdensitytilde,currgradtilde,_,gradminus = buildtree(thetaminus,rminus,u,vj,j,epsilon,theta0,r0,Q,initialdr,gradminus)

            else:
                _,_,thetaplus,rplus,thetatilde,ntilde,stilde,alfa,nalfa,currdensitytilde,currgradtilde,gradplus,_ = buildtree(thetaplus, rplus, u,vj,j,epsilon,theta0,r0,Q,initialdr,gradplus)

            if (stilde ==1):
                if(np.random.rand() < ntilde/n):
                    if(cm == False):
                        theta[:, i] = np.ravel(thetatilde)
                    theta0 = thetatilde
                    currdensity= currdensitytilde
                    currgrad = currgradtilde

            n = n+ntilde
            s = stilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus) >= 0)
            j = j +1

        if(i <=Madapt):
            Hhat = (1.0-1.0/(i+t0))*Hhat + 1.0/(i+t0)*(delta - alfa/nalfa)
            epsilon = np.exp(myy -np.sqrt(i)/gamma*Hhat)
            epsilonhat = np.exp(i**(-kappa)*np.log(epsilon)+(1.0-i**(-kappa))*np.log(epsilonhat))

        else:
            epsilon = epsilonhat
            if (cm):
                cmestimate = 1.0 / ((i-Madapt)) * ((i-Madapt-1) * cmestimate + theta0)

    print ("Epsilon: ", epsilon)
    if(cm == False):
        return theta[:,Madapt:]
    else:
        return cmestimate


        
        
     
    #a = abs(a);
    #a = min(a,3);


'''
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
@cython.cdivision(True)
def mwg_tv(N,Nadapt,Q, x0, sampsigma=1.0,cmesti=False):
    if (Nadapt >= N):
        raise Exception('Nadapt <= N.')
    
    cdef int adapt = Nadapt
    y = Q.y
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    regalpha = Q.a
    lhsigma = Q.s2
    samplebeta = Q.b
    cdef bint cm = cmesti
    
    np.random.seed(1)
    dimnumpy = x0.shape[0]
    cdef int dim = dimnumpy
    x = x0
    np.random.seed(1)
    
    if not isinstance(M, sp.csc.csc_matrix):
        M = csc_matrix(M)
    
    if not isinstance(Lx, sp.csc.csc_matrix):
        Lx = csc_matrix(Lx)
        
    if not isinstance(Ly, sp.csc.csc_matrix):
        Ly = csc_matrix(Ly)    

    cdef double alpha = regalpha
    cdef double samplingsigma = sampsigma
    cdef double beta = samplebeta
    cdef double Ci = 1.0/lhsigma**2.0
    
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
        return np.ravel(np.exp( -0.5*Ci*(M @ xf - y).T  @ (M @ xf - y)))*np.ravel(np.exp(-alpha*np.sum(np.abs(Lx @ xf))))*np.ravel(np.exp(-alpha*np.sum(np.abs(Ly@xf))))
    
    def pdff(xf):
        #return np.ravel(np.exp(-Ci * (M @ x - y).T @ (M @ x - y)))
        return (np.sum((M @ xf - y).T  @ (M @ xf - y)) ,np.sum(np.abs(Lx @ xf)) ,np.sum(np.abs(Ly @ xf)) )
    #return np.ravel(np.exp(-1.0 / 2.0 * (M@x-y).T @ Ci @ (M@x-y)))

    
    if (cm==False):
        chain = np.zeros((dim, N))
        chain[:,0] = np.ravel(x)
    else:
        chain = np.zeros((dim,1))
    
    
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
    #print(pdf(x0))
   
    cdef int i
    cdef int j
    cdef double[:, :] xv = x
    cdef double[:, :] yv = y
    cdef double new,change, change2, change3, old,currentvalue,currentmean,previousmean,currentvar
    cdef double[:, :] chainv = chain
  
    cdef int k, start, stop
    cdef double[:] acceptv,number
    chmean = np.ravel(np.copy(x0))
    chdev = np.zeros((dim,))
    cdef double[:] chmeanv = chmean
    cdef double[:]  chdevv = chdev
    
    cmest = np.copy(np.ravel(x))
    #print(x.shape,dim)
    cdef double[:] values = np.copy(np.ravel(x))
    cdef double[:] cmestimate = cmest
    
    #print(y)
    #print(w)
    #print(likelihood)
    for i in range(1,N):
        randoms = np.random.randn(dim,)
        accept = np.random.rand(dim,)
        acceptv = accept
        number = randoms
        
        with nogil:
            for j in range(0,dim):
                old = values[j]
                #old = chainv[j,i-1]
                currentvalue = old
                if (i > adapt):
                    samplingsigma = 2.4*chdevv[j] + 10**(-12)
                new = old + samplingsigma*number[j]
                #print(new)
                #new = sqrt(1.0-beta*beta)*old + beta*samplingsigma*samplingsigma*number[j]
                #print(new)

                change = 0
                change2 = 0
                change3 = 0

                start = Mptr[j]
                stop = Mptr[j+1]

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change += -(lhcompv[Mindices[k],0]- yv[Mindices[k],0])**2.0 + (Mdata[k]*(new-old) + lhcompv[Mindices[k],0] - yv[Mindices[k],0])**2.0 
                    #print(Mdata[k],yv[Mindices[k],0], lhcompv[Mindices[k],0])
                    #change += (new*Mdata[k] - yv[Mindices[k],0])**2.0 - lhcompv[Mindices[k],0]

                start = Lxptr[j]
                stop = Lxptr[j+1]    

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change2 += -fabs(prcompv[Lxindices[k],0] ) + fabs(Lxdata[k]*(new-old) + prcompv[Lxindices[k],0])

                start = Lyptr[j]
                stop = Lyptr[j+1]    

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change3 += -fabs(prcompv2[Lyindices[k],0] ) + fabs(Lydata[k]*(new-old) + prcompv2[Lyindices[k],0])    

                #xc[j,0] = new
                #dd = (likelihood+change,prior+change2)
                #ddf = (likelihood,prior)
                ratio = exp(-0.5*Ci*change -alpha*change2 - alpha*change3)
                #ratio2 = pdf(xc)/pdf(x)
                #print(ratio-ratio2)
                #print(pdf(xc)/pdf(x))
                #chain2[j,i] = ratio
                if(acceptv[j] <= ratio):
                    values[j] = new
                    currentvalue = new
                    #chainv[j,i] = new
                    likelihood = likelihood + change
                    prior = prior + change2
                    prior2 = prior2 + change3

                    start = Lxptr[j]
                    stop = Lxptr[j+1]    

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):    
                        prcompv[Lxindices[k],0] = prcompv[Lxindices[k],0] + Lxdata[k]*(new-old)

                    start = Lyptr[j]
                    stop = Lyptr[j+1]    

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):    
                        prcompv2[Lyindices[k],0] = prcompv2[Lyindices[k],0] + Lydata[k]*(new-old)    

                    start = Mptr[j]
                    stop = Mptr[j+1]

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):     
                        lhcompv[Mindices[k],0] = lhcompv[Mindices[k],0] + Mdata[k]*(new-old)

                    #x[j,0] = new

                else:
                    values[j] = old
                    #chainv[j,i] = old
                    
                if (cm==False):
                    #pass
                    chainv[j,i] = values[j]
                else:
                    if(i > adapt):
                        #cmestimate[j]  = 1.0 / ((i+1)) * ((i) * cmestimate[j] + values[j])
                        cmestimate[j]  = 1.0 / ((i-adapt)) * ((i-adapt-1) * cmestimate[j] + values[j])
                    
                previousmean = chmeanv[j]
                currentmean = 1.0/(i+1.0)*(i*previousmean+currentvalue)
                chmeanv[j] = currentmean
                currentvar = (i-1.0)/(i)*chdevv[j]*chdevv[j] + 1.0/(i+1.0)*(currentvalue-previousmean)*(currentvalue-previousmean)
                chdevv[j] = sqrt(currentvar) 
                #with gil:
                #    print(currentvar,previousmean,currentmean,currentvalue)
                #xc[j,0] = old
            #change = np.reshape(M[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #change2 = np.reshape(L[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #newvalue=np.exp((change-y+w).T@(change-y+w)*Ci*-0.5)
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
    if(cm):
        return np.reshape(cmest,(-1,1))
    else:
        return  chain


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def mwg_cauchy(N,Nadapt,Q, x0, sampsigma=1.0,cmesti=False):
    if (Nadapt >= N):
        raise Exception('Nadapt <= N.')
    
    cdef int adapt = Nadapt
    y = Q.y
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    regalpha = Q.a
    lhsigma = Q.s2
    samplebeta = Q.b
    
    cdef bint cm = cmesti
    dimnumpy = x0.shape[0]
    cdef int dim = dimnumpy
    x = x0
    np.random.seed(1)
    
    if not isinstance(M, sp.csc.csc_matrix):
        M = csc_matrix(M)
    
    if not isinstance(Lx, sp.csc.csc_matrix):
        Lx = csc_matrix(Lx)
        
    if not isinstance(Ly, sp.csc.csc_matrix):
        Ly = csc_matrix(Ly)    

    cdef double alpha = regalpha
    cdef double samplingsigma = sampsigma
    cdef double beta = samplebeta
    cdef double Ci = 1.0/lhsigma**2.0
    
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
        return np.ravel(np.exp( -0.5*Ci*(M @ xf - y).T  @ (M @ xf - y)))*np.ravel(np.prod(1.0/(alpha+np.power(Lx @ xf,2))))*np.ravel(np.prod(1.0/(alpha+np.power(Ly @ xf,2)))) 
    
    
    def pdff(xf):
        #return np.ravel(np.exp(-Ci * (M @ x - y).T @ (M @ x - y)))
        return (np.sum((M @ xf - y).T  @ (M @ xf - y)) ,np.prod(1/(alpha+np.power(Lx @ xf,2.0))) ,np.prod(1/(alpha+np.power(Ly @ xf,2.0))) )
    #return np.ravel(np.exp(-1.0 / 2.0 * (M@x-y).T @ Ci @ (M@x-y)))
    
   
    if (cm==False):
        chain = np.zeros((dim, N))
        chain[:,0] = np.ravel(x)
    else:
        chain = np.zeros((dim,1))
    
    cdef int acc = 0
    w = M@x
    w2 = Lx@x
    w3 = Ly@x
    value = np.sum((w-y).T@(w-y))
    value2 = np.prod(1/(alpha+np.power(w2,2)))
    value3 = np.prod(1/(alpha+np.power(w3,2)))
    
    cdef double[:, :] lhcompv = w
    cdef double[:, :] prcompv = w2
    cdef double[:, :] prcompv2 = w3
    cdef double likelihood = value
    cdef double prior = value2
    cdef double prior2 = value3
    #print(pdf(x0))
   
    cdef int i
    cdef int j
    cdef double[:, :] xv = x
    cdef double[:, :] yv = y
    cdef double new,change, change2, change3, old,currentvalue,currentmean,previousmean,currentvar
    cdef double[:, :] chainv = chain
  
    cdef int k, start, stop
    cdef double[:] acceptv,number
    chmean = np.ravel(np.copy(x0))
    chdev = np.zeros((dim,))
    cdef double[:] chmeanv = chmean
    cdef double[:]  chdevv = chdev
    
    cmest = np.ravel(np.copy(x))
    cdef double[:] values = np.ravel(np.copy(x))
    cdef double[:] cmestimate = cmest
    #print(y)
    #print(w)
    #print(likelihood)
    for i in range(1,N):
        randoms = np.random.randn(dim,)
        accept = np.random.rand(dim,)
        acceptv = accept
        number = randoms
        
        with nogil:
            for j in range(0,dim):

                #old = chainv[j,i-1]
                old = values[j]
                currentvalue = old
                if (i > adapt):
                    samplingsigma = 2.4*chdevv[j] + 10**(-12)
                new = old + samplingsigma*number[j]
                #print(new)
                #new = sqrt(1.0-beta*beta)*old + beta*samplingsigma*samplingsigma*number[j]
                #print(new)

                change = 0
                change2 = 1
                change3 = 1

                start = Mptr[j]
                stop = Mptr[j+1]

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change += -(lhcompv[Mindices[k],0]- yv[Mindices[k],0])**2.0 + (Mdata[k]*(new-old) + lhcompv[Mindices[k],0] - yv[Mindices[k],0])**2.0 
                    #print(Mdata[k],yv[Mindices[k],0], lhcompv[Mindices[k],0])
                    #change += (new*Mdata[k] - yv[Mindices[k],0])**2.0 - lhcompv[Mindices[k],0]

                start = Lxptr[j]
                stop = Lxptr[j+1]    

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change2 *=  (alpha+(prcompv[Lxindices[k],0])**2.0)/(alpha+(Lxdata[k]*(new-old) + prcompv[Lxindices[k],0])**2.0)

                start = Lyptr[j]
                stop = Lyptr[j+1]    

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change3 *= (alpha+(prcompv2[Lyindices[k],0])**2.0) /(alpha+(Lydata[k]*(new-old) + prcompv2[Lyindices[k],0])**2.0)

                #xc[j,0] = new
                #dd = (likelihood+change,prior+change2)
                #ddf = (likelihood,prior)
                ratio = exp(-0.5*Ci*change)*change2*change3
                #print((exp(-0.5*Ci*change),change2,change3))
                #ratio2 = pdf(xc)/pdf(x)
                #print(pdff(xc))
                #print(pdff(x))
                #print(ratio2,ratio)
                #exit(1)
                #print(ratio-ratio2)

                #chain2[j,i] = ratio
                if(acceptv[j] <= ratio):
                    values[j] = new
                    #chainv[j,i] = new
                    currentvalue = new
                    #likelihood = likelihood + change
                    #prior = prior + change2
                    #prior2 = prior2 + change3

                    start = Lxptr[j]
                    stop = Lxptr[j+1]    

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):    
                        prcompv[Lxindices[k],0] = prcompv[Lxindices[k],0] + Lxdata[k]*(new-old)

                    start = Lyptr[j]
                    stop = Lyptr[j+1]    

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):    
                        prcompv2[Lyindices[k],0] = prcompv2[Lyindices[k],0] + Lydata[k]*(new-old)    

                    start = Mptr[j]
                    stop = Mptr[j+1]

                    #for k in prange(start,stop,1,nogil=True):
                    for k in range(start,stop):     
                        lhcompv[Mindices[k],0] = lhcompv[Mindices[k],0] + Mdata[k]*(new-old)

                    #x[j,0] = new

                else:
                    values[j] = old
                    #chainv[j,i] = old
                    
                if (cm==False):
                    #pass
                    chainv[j,i] = values[j]
                else:
                    if(i > adapt):
                        #cmestimate[j]  = 1.0 / ((i+1)) * ((i) * cmestimate[j] + values[j])
                        cmestimate[j]  = 1.0 / ((i-adapt)) * ((i-adapt-1) * cmestimate[j] + values[j])
                
                previousmean = chmeanv[j]
                currentmean = 1.0/(i+1.0)*(i*previousmean+currentvalue)
                chmeanv[j] = currentmean
                currentvar = (i-1.0)/(i)*chdevv[j]*chdevv[j]+ 1.0/(i+1.0)*(currentvalue-previousmean)*(currentvalue-previousmean)
                chdevv[j] = sqrt(currentvar) 
                
                    #xc[j,0] = old
            #change = np.reshape(M[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #change2 = np.reshape(L[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            #newvalue=np.exp((change-y+w).T@(change-y+w)*Ci*-0.5)
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
    if(cm):
        return np.reshape(cmest,(-1,1))
    else:
        return  chain
