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
from tqdm import tqdm
import sys

# Class for passing extra arguments for functions.
# M, Lx and Ly refer to matrices, y to a measurement vector, s2 to sigma squared. Variables a andb are used
# as regularization parameters.  Logdensity is points to a given log-PDF function  and gradi to its gradient.
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

# Logarithm of overall posterior with Cauchy difference prior and  traditional Gaussian likelihood. S2 is the variance of the likelihood. Alpha is the constant term in the Cauchy difference prior denominator.
@cython.cdivision(True)
@cython.boundscheck(False) 
@cython.wraparound(False)        
def tfun_cauchy( x,Q):
    x = np.reshape(x, (-1, 1))
    Mxy = Q.M.dot(x) - Q.y
    Lxx = Q.Lx.dot(x)
    Lyx = Q.Ly.dot(x)
    alpha = Q.a
    return   -0.5/Q.s2*Mxy.T.dot( Mxy) - np.sum(np.log(alpha + np.multiply(Lxx,Lxx))) - np.sum(
        np.log(alpha + np.multiply(Lyx,Lyx))) 
    
         

# Logarithm of overall posterior with Total Variation prior and  traditional Gaussian likelihood. S2 is the variance of the likelihood. Alpha is the regularization parameter.
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


# Logarithm of overall posterior with Tikhonov (Gaussian difference) prior and  traditional Gaussian likelihood. S2 is the variance of the likelihood. Alpha is the regularization parameter.
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
      

# Gradient of the log-Tikhonov.
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

# Gradient of the log-TV.
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
    cdef double [:] Lydata = Ly.data
    cdef int [:]Lyindices = Ly.indices
    cdef int [:]Lyptr = Ly.indptr
    (ro,co) = Ly.shape
    cdef int row = ro
    cdef int col = co
    cdef double alfa = a
    cdef double beta = b 

    Mxy = M.dot(x) - y
    cdef double [:,:] Lxx = Lx.dot(x)
    cdef double [:,:] Lyx = Ly.dot(x)
    
    # Likelihood part.
    gr = -1.0/s2  * (M.T).dot(Mxy) 
    cdef double [:,:] grv = gr
    cdef int i,j
    cdef double s
    
    # Calculation of the prior part.
    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - Lxdata[j]*Lxx[Lxindices[j],0]/sqrt(Lxx[Lxindices[j],0]*Lxx[Lxindices[j],0] + beta)

        grv[i,0] = grv[i,0] + alfa*s

    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - Lydata[j] * Lyx[Lyindices[j],0] / sqrt(Lyx[Lyindices[j],0] *Lyx[Lyindices[j],0] + beta)

        grv[i, 0] = grv[i, 0] + alfa * s 
    return gr


# Gradient of the log-Cauchy.
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
    cdef double [:] Lydata = Ly.data
    cdef int [:]Lyindices = Ly.indices
    cdef int [:]Lyptr = Ly.indptr
    (ro,co) = Ly.shape
    cdef int row = ro
    cdef int col = co
    cdef double alfa = a

    Mxy = M.dot(x) - y
   
    cdef double [:,:] Lxx = Lx.dot(x)

    cdef double [:,:] Lyx = Ly.dot(x)
    
    # Likelihood part.
    gr = -1.0/s2  * (M.T).dot(Mxy)
    cdef double [:,:] grv = gr
    cdef int i,j
    cdef double s


    # Prior part.
    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - 2*Lxx[Lxindices[j],0]/(alfa+Lxx[Lxindices[j],0]*Lxx[Lxindices[j],0])*Lxdata[j]

        grv[i,0] = grv[i,0] + s

    for i in prange(col,nogil=True):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - 2 * Lyx[Lyindices[j],0] / (alfa + Lyx[Lyindices[j],0] *Lyx[Lyindices[j],0])*Lydata[j]

        grv[i, 0] = grv[i, 0] +   s
    return gr


# Leapfrog function for HMC.
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def leapfrog(theta,r,epsilon,Q,currgrad):
    r = r + epsilon*0.5*currgrad
    theta = theta + epsilon*r
    currdensity = Q.logdensity(theta,Q)
    currgrad = Q.gradi(theta, Q)
    r = r + epsilon*0.5*currgrad
    return (theta,r,currdensity,currgrad)


# Function, which finds initial step size for HMC.
def initialeps(theta,Q,currdensity,currgrad):
    def totalenergy( r, cd):
        energy = (cd - 0.5 * np.dot(r.T, r))
        return energy
    eps = 1.0
    r = np.random.randn(theta.shape[0],1)
    (theta2,r2,currdensity2,currgrad2) = leapfrog(theta,r,eps,Q,currgrad)
    a = 2.0*(np.exp(totalenergy(r2,currdensity2) - totalenergy(r,currdensity)) > 0.5) -1.0

    while  a * (totalenergy(r2,currdensity2) - totalenergy(r,currdensity)) > -a * np.log(2):
        eps = 2.0**(a)*eps
        (theta2,r2,currdensity2,currgrad2) = leapfrog(theta,r,eps,Q,currgrad)
    print(eps)
    return eps


# Buildtree-function is the core of HMC-NUTS method.
# This function is a modified version of Morgan Fouesneau's one (MIT license):
# https://github.com/mfouesneau/NUTS/blob/master/nuts/nuts.py
# Passing the present step's gradient as an input (currgrad) argument saves calls to gradient function. 
# That also requires using extra output arguments for different branches.  
# Using logu as an function input argument instead of u helps to prevent overflows. 
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)  
def buildtree(theta,r,logu,v,j,epsilon,theta0,r0,Q,initialdr,currgrad):
    if (j==0):
        thetatilde, rtilde, currdensitytilde,currgradtilde = leapfrog(theta,r,v*epsilon,Q,currgrad)
        diff = np.min(np.array([np.exp(currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) -initialdr),1]))
        ntilde = int(logu <= (currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        stilde = int(logu < (1000.0+currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        return thetatilde,rtilde,thetatilde,rtilde,thetatilde,ntilde,stilde,diff,1,currdensitytilde,currgradtilde,currgradtilde,currgradtilde

    else:
        thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde,currdensitytilde,currgradtilde,gradplus,gradminus = buildtree(theta,r,logu,v,j-1,epsilon,theta0,r0,Q,initialdr,currgrad)

        if(stilde == 1):
            if(v == -1):
                thetaminus, rminus, _, _, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde,currdensitytildetilde,currgradtildetilde,_,gradminus = buildtree(
                    thetaminus, rminus, logu, v, j - 1, epsilon, theta0, r0,Q,initialdr,gradminus)
            else:
                _, _, thetaplus, rplus, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde,currdensitytildetilde,currgradtildetilde,gradplus,_ = buildtree(
                    thetaplus, rplus, logu, v, j - 1, epsilon, theta0, r0,Q,initialdr,gradplus)
            if(np.random.rand() <= ntildetilde/np.max(np.array([ntilde + ntildetilde,1]))):
                thetatilde = thetatildetilde
                currdensitytilde = currdensitytildetilde
                currgradtilde = currgradtildetilde

            alfatilde = alfatilde + alfatildetilde
            nalfatilde = nalfatilde + nalfatildetilde
            stilde = stildetilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus)>=0)
            ntilde = ntilde + ntildetilde
        return thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde,currdensitytilde,currgradtilde,gradplus,gradminus


# The main HMC method. 
# M is the number of samples to be generated, theta0 is the initial parameter vector, Q is the argumentpack, Madapt is the number of dual averaging steps, 
# de is the target accept ratio. Gamma, t0 and kappa are adaptation parameters. If epsilonwanted is not None, its value is used as an step size and adaptation is not made. 
# Furthermore, insatnce of class Q includes the target posterior and its derivative.
# Cm is a boolean value, if it's True, only conditional mean is calculated and not the whole chain is returned.
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def hmc(M,theta0,Q,Madapt,de=0.6,gamma=0.05,t0=10.0,kappa=0.75,epsilonwanted=None,cmonly=False,thinning=1):
    bar = tqdm(total=M,file=sys.stdout)
    theta0 = np.reshape(theta0,(-1,1))
    dim = theta0.shape[0]
    if (cmonly == False):
        theta = np.zeros((dim, M//thinning + 1))
        theta[:, 0] = np.ravel(theta0)
    delta = de
    currdensity = Q.logdensity(theta0, Q)
    currgrad = Q.gradi(theta0, Q)
    if (epsilonwanted is None):
        epsilon = initialeps(theta0,Q,currdensity,currgrad)
        epsilonhat = 1.0
    else:
        epsilon = epsilonwanted
        epsilonhat = epsilonwanted
    myy = np.log(10*epsilon)
    Hhat = 0.0
    cmestimate = np.zeros((dim,1))
    if (Madapt >= M):
        raise Exception('Madapt <= M.')
    if (de < 0.05 or de > 0.999999):
        raise Exception('Delta is not within reasonable range.')

    for i in range(1,M+1):
        bar.update(1)
        r0 = np.random.randn(dim, 1)
        logu = (currdensity - 0.5*np.dot(r0.T,r0))+np.log(np.random.rand())
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
        if(cmonly == False and (i%thinning == 0)):
            theta[:, i//thinning] = np.ravel(theta0)
  
        while (s==1):
            vj = np.random.choice(np.array([-1,1]))
            if (vj ==-1):
                thetaminus,rminus,_,_,thetatilde,ntilde,stilde,alfa,nalfa,currdensitytilde,currgradtilde,_,gradminus = buildtree(thetaminus,rminus,logu,vj,j,epsilon,theta0,r0,Q,initialdr,gradminus)

            else:
                _,_,thetaplus,rplus,thetatilde,ntilde,stilde,alfa,nalfa,currdensitytilde,currgradtilde,gradplus,_ = buildtree(thetaplus, rplus, logu,vj,j,epsilon,theta0,r0,Q,initialdr,gradplus)

            if (stilde ==1):
                if(np.random.rand() < ntilde/n):
                    if(cmonly == False and (i%thinning == 0)):
                        theta[:, i//thinning] = np.ravel(thetatilde)
                    theta0 = thetatilde
                    currdensity= currdensitytilde
                    currgrad = currgradtilde

            n = n+ntilde
            s = stilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus) >= 0)
            j = j +1
        
       
        if(i <=Madapt):
            if (epsilonwanted is None):
                Hhat = (1.0-1.0/(i+t0))*Hhat + 1.0/(i+t0)*(delta - alfa/nalfa)
                epsilon = np.exp(myy -np.sqrt(i)/gamma*Hhat)
                epsilonhat = np.exp(i**(-kappa)*np.log(epsilon)+(1.0-i**(-kappa))*np.log(epsilonhat))

        else:
            epsilon = epsilonhat
            cmestimate = 1.0 / ((i-Madapt)) * ((i-Madapt-1) * cmestimate + theta0)
    
    bar.close()
    print ("Final epsilon: " +  str(np.ravel(epsilon)))
    if(cmonly == False):
        return cmestimate,theta
    else:
        return cmestimate,None
     


# Metropolis within-Gibbs function for TV prior and Gaussian likelihood.
# N is the number of samples, Nadapt is the number of SCAM steps, Q is the extra argument instance, 
# x0 is the initial parameter vector, sampsigma is the initial proposal step size.
# Cmesti is the boolean which determines whether the whole chain or CM estimate is returned.
@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def mwg_tv(N,Nadapt,Q, x0, sampsigma=1.0,cmonly=False,thinning=10):
    bar = tqdm(total=N,file=sys.stdout)
    if (Nadapt >= N):
        raise Exception('Nadapt <= N.')
    
    cdef int adapt = Nadapt
    cdef int thin = thinning
    y = Q.y
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    regalpha = Q.a
    lhvariance = Q.s2
    samplebeta = Q.b
    cdef bint cm = cmonly
    
    np.random.seed(1)
    dimnumpy = x0.shape[0]
    cdef int dim = dimnumpy
    x = x0
    
    if not isinstance(M, sp.csc.csc_matrix):
        M = csc_matrix(M)
    
    if not isinstance(Lx, sp.csc.csc_matrix):
        Lx = csc_matrix(Lx)
        
    if not isinstance(Ly, sp.csc.csc_matrix):
        Ly = csc_matrix(Ly)    

    cdef double alpha = regalpha
    cdef double samplingsigma = sampsigma
    cdef double beta = samplebeta
    cdef double Ci = 1.0/lhvariance
    
    cdef double[:] Mdata =  M.data
    cdef int[:] Mindices =  M.indices
    cdef int[:] Mptr =  M.indptr
    
    cdef double[:] Lxdata =  Lx.data
    cdef int[:] Lxindices =  Lx.indices
    cdef int[:] Lxptr =  Lx.indptr
    
    cdef double[:] Lydata =  Ly.data
    cdef int[:] Lyindices =  Ly.indices
    cdef int[:] Lyptr =  Ly.indptr
 
    
    if (cm==False):
        chain = np.zeros((dim, N//thin+1))
        chain[:,0] = np.ravel(x)
    else:
        chain = np.zeros((dim,1))
    
    
    cdef int acc = 0
    w = M@x
    w2 = Lx@x
    w3 = Ly@x
    
    cdef double[:, :] lhcompv = w
    cdef double[:, :] prcompv = w2
    cdef double[:, :] prcompv2 = w3

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
    cdef double[:] values = np.copy(np.ravel(x))
    cdef double[:] cmestimate = cmest
    
    for i in range(1,N+1):
        randoms = np.random.randn(dim,)
        accept = np.random.rand(dim,)
        acceptv = accept
        number = randoms
        bar.update(1)
        with nogil:
            for j in range(0,dim):
                old = values[j]
               
                currentvalue = old
                if  (i> 20) and (i <= adapt):
                    samplingsigma = 1.542724*chdevv[j] + 10**(-6)
                    
                new = old + samplingsigma*number[j]


                change = 0
                change2 = 0
                change3 = 0

                start = Mptr[j]
                stop = Mptr[j+1]

                #for k in prange(start,stop,1,nogil=True):
                for k in range(start,stop):    
                    change += -(lhcompv[Mindices[k],0]- yv[Mindices[k],0])**2.0 + (Mdata[k]*(new-old) + lhcompv[Mindices[k],0] - yv[Mindices[k],0])**2.0 
 

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

             
                ratio = exp(-0.5*Ci*change -alpha*change2 - alpha*change3)
                if(acceptv[j] <= ratio):
                    values[j] = new
                    currentvalue = new
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
                    
                if ((cm==0) and (i%thin == 0)):
                    chainv[j,i//thin] = values[j]
                    
                #else:
                if(i > adapt):
                    #cmestimate[j]  = 1.0 / ((i+1)) * ((i) * cmestimate[j] + values[j])
                    cmestimate[j]  = 1.0 / ((i-adapt)) * ((i-adapt-1) * cmestimate[j] + values[j])

                if (i <= adapt):
                    previousmean = chmeanv[j]
                    currentmean = 1.0/(i+1.0)*(i*previousmean+currentvalue)
                    chmeanv[j] = currentmean
                    currentvar = (i-1.0)/(i)*chdevv[j]*chdevv[j] + 1.0/(i+1.0)*(currentvalue-previousmean)*(currentvalue-previousmean)
                    chdevv[j] = sqrt(currentvar) 
                
        
    bar.close()
    if(cm):
        return np.reshape(cmest,(-1,1)),None
    else:
        return  np.reshape(cmest,(-1,1)),chain

# Metropolis within-Gibbs function for Cauchy prior and Gaussian likelihood.
# N is the number of samples, Nadapt is the number of SCAM steps, Q is the extra argument instance, 
# x0 is the initial parameter vector, sampsigma is the initial proposal step size.
# Cmesti is the boolean which determines whether the whole chain or CM estimate is returned.
@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def mwg_cauchy(N,Nadapt,Q, x0, sampsigma=1.0,cmonly=False,thinning=10):
    bar = tqdm(total=N,file=sys.stdout)
    if (Nadapt >= N):
        raise Exception('Nadapt <= N.')
    
    cdef int adapt = Nadapt
    cdef int thin = thinning
    y = Q.y
    M = Q.M
    Lx = Q.Lx
    Ly = Q.Ly
    regalpha = Q.a
    lhvariance = Q.s2
    samplebeta = Q.b
    
    cdef int cm = cmonly
    dimnumpy = x0.shape[0]
    cdef int dim = dimnumpy
    x = x0
    
    if not isinstance(M, sp.csc.csc_matrix):
        M = csc_matrix(M)
    
    if not isinstance(Lx, sp.csc.csc_matrix):
        Lx = csc_matrix(Lx)
        
    if not isinstance(Ly, sp.csc.csc_matrix):
        Ly = csc_matrix(Ly)    

    cdef double alpha = regalpha
    cdef double samplingsigma = sampsigma
    cdef double beta = samplebeta
    cdef double Ci = 1.0/lhvariance
    
    cdef double[:] Mdata =  M.data
    cdef int[:] Mindices =  M.indices
    cdef int[:] Mptr =  M.indptr
    
    cdef double[:] Lxdata =  Lx.data
    cdef int[:] Lxindices =  Lx.indices
    cdef int[:] Lxptr =  Lx.indptr
    
    cdef double[:] Lydata =  Ly.data
    cdef int[:] Lyindices =  Ly.indices
    cdef int[:] Lyptr =  Ly.indptr
 
   
    if (cm==0):
        chain = np.zeros((dim, N//thin+1))
        chain[:,0] = np.ravel(x)
    else:
        chain = np.zeros((dim,1))
    
    cdef int acc = 0
    w = M@x
    w2 = Lx@x
    w3 = Ly@x
   
    cdef double[:, :] lhcompv = w
    cdef double[:, :] prcompv = w2
    cdef double[:, :] prcompv2 = w3
   
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

    for i in range(1,N+1):
        bar.update(1)
        randoms = np.random.randn(dim,)
        accept = np.random.rand(dim,)
        acceptv = accept
        number = randoms
        
        with nogil:
            for j in range(0,dim):

                old = values[j]
                currentvalue = old
                if  (i> 20) and (i <= adapt):
                    samplingsigma = 1.542724*chdevv[j] + 10**(-6)
                new = old + samplingsigma*number[j]


                change = 0
                change2 = 1
                change3 = 1

                start = Mptr[j]
                stop = Mptr[j+1]

                #for k in prange(start,stop,1):
                for k in range(start,stop):    
                    change += -(lhcompv[Mindices[k],0]- yv[Mindices[k],0])**2.0 + (Mdata[k]*(new-old) + lhcompv[Mindices[k],0] - yv[Mindices[k],0])**2.0 

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


                ratio = exp(-0.5*Ci*change)*change2*change3
                if(acceptv[j] <= ratio):
                    values[j] = new
                    currentvalue = new
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

                    #for k in prange(start,stop,1):
                    for k in range(start,stop):     
                        lhcompv[Mindices[k],0] = lhcompv[Mindices[k],0] + Mdata[k]*(new-old)


                else:
                    values[j] = old
                    
                if ((cm==0) and (i%thin == 0)):
                    chainv[j,i//thin] = values[j]
                #else:
                if(i > adapt):
                    #cmestimate[j]  = 1.0 / ((i+1)) * ((i) * cmestimate[j] + values[j])
                    cmestimate[j]  = 1.0 / ((i-adapt)) * ((i-adapt-1) * cmestimate[j] + values[j])
                
                if (i <= adapt):
                    previousmean = chmeanv[j]
                    currentmean = 1.0/(i+1.0)*(i*previousmean+currentvalue)
                    chmeanv[j] = currentmean
                    currentvar = (i-1.0)/(i)*chdevv[j]*chdevv[j]+ 1.0/(i+1.0)*(currentvalue-previousmean)*(currentvalue-previousmean)
                    chdevv[j] = sqrt(currentvar) 
    bar.close()     
    if(cm):
        return np.reshape(cmest,(-1,1)),None
    else:
        return  np.reshape(cmest,(-1,1)),chain
