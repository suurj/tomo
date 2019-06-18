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

def tv_grad(M, Lx, Ly, y, x, s2, alfa, beta):
    Lxdata =  Lx.data
    Lxindices = Lx.indices
    Lxptr = Lx.indptr

    Lydata = Ly.data
    Lyindices = Ly.indices
    Lyptr = Ly.indptr
    (row,col) = Ly.shape

    # Mxy = np.dot(M,x)-y
    Mxy = M.dot(x) - y
    # Lxx = np.dot(Lx,x)
    Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Ly.dot(x)

    #common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -s2 * 2.0 * (M.T).dot(Mxy) 


    #exit(1)
    for i in range(col):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - Lxdata[j]*Lxx[Lxindices[j]]/np.sqrt(Lxx[Lxindices[j]]**2 + beta)

        #exit(0)
        gr[i,0] = gr[i,0] + alfa*s

    for i in range(col):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - Lydata[j] * Lyx[Lyindices[j]] / np.sqrt(Lyx[Lyindices[j]] ** 2 + beta)

        # exit(0)
        gr[i, 0] = gr[i, 0] + alfa * s 
    return gr

def cauchy_grad(M, Lx, Ly, y, x, s2, alfa, beta):
    Lxdata =  Lx.data
    Lxindices = Lx.indices
    Lxptr = Lx.indptr

    Lydata = Ly.data
    Lyindices = Ly.indices
    Lyptr = Ly.indptr
    (row,col) = Ly.shape

    # Mxy = np.dot(M,x)-y
    Mxy = M.dot(x) - y
    # Lxx = np.dot(Lx,x)
    Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Ly.dot(x)

    #common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -s2 * 2.0 * (M.T).dot(Mxy)


    #exit(1)
    for i in range(col):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - 2*Lxx[Lxindices[j]]/(alfa+Lxx[Lxindices[j]]**2)*Lxdata[j]

        #exit(0)
        gr[i,0] = gr[i,0] + s

    for i in range(col):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - 2 * Lyx[Lyindices[j]] / (alfa + Lyx[Lyindices[j]] ** 2)*Lydata[j]

        # exit(0)
        gr[i, 0] = gr[i, 0] +   s
    return gr

@cython.boundscheck(False) 
@cython.wraparound(False)
def gradi(x,A):
    return -0.5*2*((A[0].T).dot(A[0])).dot(x)
    #*np.exp(-0.5*np.dot(res.T,res))
    #return np.reshape(-np.matmul(np.matmul(A[0].T,A[0]),x),(-1,1))


def logdensity(theta,A):
    #print(A[0].dot(theta))
    #print(-1* np.dot(A[0].dot(theta).T, A[0].dot(theta)),theta)
    return -0.5* np.dot(A[0].dot(theta).T, A[0].dot(theta))
    #return 2 * np.dot((np.dot(A[0], theta)).T , (np.dot(A[0], theta)))
    #return np.exp(-0.5*(np.dot(A,theta)).T@(np.dot(A,theta)) - 0.5 * np.dot(r.T, r))
    #return np.exp(np.log(pdf(theta)) - 0.5 * np.dot(r.T, r))
    #return np.exp(np.log(pdf(theta))-0.5*np.dot(r.T,r))


def leapfrog(theta,r,epsilon,A):
    r = r + (epsilon/2.0)*gradi(theta,A)
    theta = theta + epsilon*r
    r = r + (epsilon/2.0)*gradi(theta,A)
    #print(theta)
    return (theta,r)


def initialeps(theta,A):
    def totalenergy(x, r, A):
        # return np.exp(-1/2*x.T@Q@x)
        # return np.exp(-1/2*np.sum(np.power(R@x,2)))
        energy = (logdensity(x, A) - 0.5 * np.dot(r.T, r))
        # if (x[0] > 10):
        #    print(x)
        return energy
    eps = 1.0
    r = np.random.randn(theta.shape[0],1)
    (theta2,r2) = leapfrog(theta,r,eps,A)
    a = 2.0*(np.exp(totalenergy(theta2,r2,A) - totalenergy(theta,r,A)) > 0.5) -1.0
    print(totalenergy(theta2,r2,A) - totalenergy(theta,r,A),a)

    while  a * (totalenergy(theta2,r2,A) - totalenergy(theta,r,A)) > -a * np.log(2):
        eps = 2.0**(a)*eps
        (theta2,r2) = leapfrog(theta,r,eps,A)
    return eps


def buildtree(theta,r,u,v,j,epsilon,theta0,r0,A):
    if (j==0):
        thetatilde, rtilde = leapfrog(theta,r,v*epsilon,A)
        ldensitytilde = logdensity(thetatilde,A)
        ldensity = logdensity(theta0,A)
        logu = np.log(u)
        diff =np.exp(ldensitytilde - 0.5*np.dot(rtilde.T,rtilde) - ldensity + 0.5*np.dot(r0.T,r0))

        ntilde = float(logu <= (ldensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        stilde = float(logu < (1000.0+ldensitytilde - 0.5*np.dot(rtilde.T,rtilde) )  )
        return thetatilde,rtilde,thetatilde,rtilde,thetatilde,ntilde,stilde,np.min(np.array([1,diff])),1

    else:
        thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde = buildtree(theta,r,u,v,j-1,epsilon,theta0,r0,A)

        if(stilde == 1):
            if(v == -1):
                thetaminus, rminus, _, _, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde = buildtree(
                    thetaminus, rminus, u, v, j - 1, epsilon, theta0, r0,A)
            else:
                _, _, thetaplus, rplus, thetatildetilde, ntildetilde, stildetilde, alfatildetilde, nalfatildetilde = buildtree(
                    thetaplus, rplus, u, v, j - 1, epsilon, theta0, r0,A)
            if(np.random.rand() <= ntildetilde/np.max(np.array([ntilde + ntildetilde,1]))):
                thetatilde = thetatildetilde

            alfatilde = alfatilde + alfatildetilde
            nalfatilde = nalfatilde + nalfatildetilde
            stilde = stildetilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus))
            ntilde = ntilde + ntildetilde


        return thetaminus,rminus,thetaplus,rplus,thetatilde,ntilde,stilde,alfatilde,nalfatilde
#@profile

def hmc(M,x):
    A = (np.eye(3)*1,None)
    x = np.reshape(x,(-1,1))
    dim = x.shape[0]
    theta = np.zeros((dim, M))
    theta0 = x
    theta[:, 0] = np.ravel(theta0)
    delta = 0.60
    epsilon = initialeps(theta0,A)
    myy = np.log(10*epsilon)
    epsilonhat = 1.0
    Hhat = 0.0
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    Madapt = int(0.07*M)

    for i in range(1,M):

        r0 = np.random.randn(dim, 1)
        u = np.exp(logdensity(theta[:,i-1],A) - 0.5*np.dot(r0.T,r0))*np.random.rand()
        thetaminus = np.reshape(theta[:,i-1],(-1,1))
        thetaplus = np.reshape(theta[:,i-1],(-1,1))
        rminus = r0
        rplus = r0
        j = 0
        n = 1.0
        s = 1.0
        theta[:, i] = theta[:, i - 1]
        #exit(1)
        while (s==1):
            vj = np.random.choice(np.array([-1,1]))
            if (vj ==-1):
                thetaminus,rminus,_,_,thetatilde,ntilde,stilde,alfa,nalfa = buildtree(thetaminus,rminus,u,vj,j,epsilon,theta[:,i-1],r0,A)
            else:
                _,_,thetaplus,rplus,thetatilde,ntilde,stilde,alfa,nalfa = buildtree(thetaplus, rplus, u,vj,j,epsilon,theta[:,i-1],r0,A)

            if (stilde ==1):
                if(np.random.rand() < ntilde/n):
                    theta[:, i] = np.ravel(thetatilde)

            n = n+ntilde
            s = stilde*(np.dot((thetaplus -thetaminus).T,rminus) >= 0)*(np.dot((thetaplus -thetaminus).T,rplus) >= 0)
            j = j +1

        if(i <=Madapt):
            Hhat = (1.0-1.0/(i+t0))*Hhat + 1.0/(i+t0)*(delta - alfa/nalfa)
            epsilon = np.exp(myy -np.sqrt(i)/gamma*Hhat)
            epsilonhat = np.exp(i**(-kappa)*np.log(epsilon)+(1.0-i**(-kappa))*np.log(epsilonhat))
        else:
            epsilon = epsilonhat

    return theta[:,Madapt:]

@cython.boundscheck(False) 
#@cython.wraparound(False)
@cython.cdivision(True) 
def  radonmatrix(size,theta):
    cdef cpplist[int] row
    cdef cpplist[int] col
    cdef cpplist[double] data
    
    cdef int T = theta.shape[0]
    cdef int N = size
    cdef double dx = 1
    cdef double dy = 1 
    cdef int M = N
    cdef int R = math.ceil(M_SQRT2*N)
    #cdef int R = Rr
    cdef double xmin = -(N-1.0)/2.0
    cdef double dp = 2.0*M_SQRT2*abs(xmin)/(R-1.0)
    cdef double pmin = -(R-1.0)/2.0*dp
    cdef double tmin = theta[0]
    cdef double ymin = xmin
    cdef double pmax = (R-1.0)/2.0*dp
    cdef double tmax = theta[-1]
    cdef double tt,ray,dt
    cdef int t,m,n,r,i
    if (T == 1):
        dt = 0
    else:
        dt = theta[1]-theta[0]
    
    start = time.time() 
    with nogil:                     
        for r in range (0,R):
            for t in range (0,T):
                tt = tmax - t*dt
                for n in range (0,N):
                    for m in range( 0,M):
                        #psi = pi/dx*(pmin+r*dp-(xmin+m*dx)*cos(tt)-(ymin+n*dy)*sin(tt))
                        #A(r*T + t+1, n*M+m+1) = dx/2 * g(2*(pmin+r*dp -(xmin+m*dx)*cos(tt)-(ymin+n*dy)*sin(tt) )/dx,tt)
                        #ray = dx/2.0 * gs(2.0*(pmax-r*dp -(xmin+m*dx)*cos(tt)-(ymin+n*dy)*sin(tt) )/dx,tt)
                        ray = dx/2.0 * gs(2.0*(pmax-r*dp+dp/4.0 -(xmin+m*dx)*cos(tt+dt/4.0)-(ymin+n*dy)*sin(tt+dt/4.0) )/dx,tt+dt/4.0)
                        ray = ray + dx/2.0 * gs(2.0*(pmax-r*dp+dp/4.0 -(xmin+m*dx)*cos(tt-dt/4.0)-(ymin+n*dy)*sin(tt-dt/4.0) )/dx,tt-dt/4.0)
                        ray = ray + dx/2.0 * gs(2.0*(pmax-r*dp-dp/4.0 -(xmin+m*dx)*cos(tt+dt/4.0)-(ymin+n*dy)*sin(tt+dt/4.0) )/dx,tt+dt/4.0)
                        ray = ray + dx/2.0 * gs(2.0*(pmax-r*dp-dp/4.0 -(xmin+m*dx)*cos(tt-dt/4.0)-(ymin+n*dy)*sin(tt-dt/4.0) )/dx,tt-dt/4.0)
                        ray = ray/4.0
                        if(ray > 0.0):
                            row.push_back(r*T+t)
                            col.push_back(n*M+m)
                            data.push_back(ray)
                    #A(r*T + t+1, n*M+m+1) = 

            
    print(time.time()-start)     
     
                     

    #push_back = temp.push_back
    #for x in range(5):
    #    row.push_back(x)
    #    col.push_back(x+1)
    #    data.push_back(2.0)

    cdef int Nel = row.size()
    coo_row  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_rowv  = coo_row
    coo_col  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_colv  = coo_col
    coo_data = np.zeros((Nel,))
    cdef double [:] coo_datav  = coo_data

    #front = temp.front()
    #pop_front = temp.pop_front()
    for i in range(Nel):
        coo_rowv[i] = row.front()
        row.pop_front()
        
        coo_colv[i] = col.front()
        col.pop_front()

        coo_datav[i] = data.front()
        data.pop_front()

    #return (coo_data,coo_row,coo_col)
    #dd = np.array([2,3,4])
    #cc = np.array([0,1,0.0])
    #rr = np.array([1,0,0])
    #radonM = coo_matrix((dd, (rr, cc)), shape=(2,2))
    
    radonM = coo_matrix((coo_data, (coo_row, coo_col)), shape=(R*T,N*N))
    radonM = csc_matrix(radonM)
    
    return radonM 

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True) 
cdef double gs(double p,double t) nogil:
    #cdef double pi = np.pi
    #cdef double sqrt2 = np.sqrt(2.0)
    cdef double x1m
    cdef double x1
    cdef double y1
    if (p<0):
       p = -p
   
    t = (t%(M_PI/2.0))
    if(t >= M_PI/4.0):
        t = M_PI/2.0-t
  
    #if (t > pi/2)
    #    t = pi/2 - mod(t,pi/4);
    #end
    
    if( p > M_SQRT2):
        a = 0
        return a
    else:
        x1m = p/cos(t) + tan(t)
        x1 = p/cos(t) - tan(t)
        y1 = p/sin(t) - 1.0/tan(t)
        
   
        if (x1 < 1.0 and x1m  < 1.0):
            #a = 2/cos(t);
            a = sqrt(4.0+(x1-x1m)**2.0)
            return a
            #disp("TOO")
            
        elif (x1 < 1.0 and x1m  > 1.0):
            a = sqrt((1.0-x1)**2.0 + (1.0-y1)**2.0)
            return a
            #disp('RRR')
            
        elif (x1 >=1.0):
            a = 0.0
            return a
            
        else:
            return -9.0
        
        
     
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
'''    
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
def mwg_tv(M, Lx, Ly,  y, x0,N, regalpha=1, samplebeta=0.3, sampsigma=1,lhsigma=1):
    dimnumpy = y.shape[0]
    cdef int dim = dimnumpy
    x = x0
    xc = np.copy(x)
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
    cdef double Ci = 1.0/lhsigma
    
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

    
    chain = np.zeros((dim, N))
    chain[:,0] = np.ravel(x)
    chain2 = np.copy(chain)
    
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
        
        with nogil:
            for j in range(0,dim):

                old = chainv[j,i-1]
                #new = old + samplingsigma*number[j]
                #print(new)
                new = sqrt(1.0-beta*beta)*old + beta*samplingsigma*samplingsigma*number[j]
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
                    chainv[j,i] = new
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
                    chainv[j,i] = old
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
    return  chain


@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def mwg_cauchy(M, Lx, Ly,  y, x0,N, regalpha=1, samplebeta=0.3, sampsigma=1,lhsigma=1):
    dimnumpy = y.shape[0]
    cdef int dim = dimnumpy
    x = x0
    xc = np.copy(x)
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
    cdef double Ci = 1.0/lhsigma
    
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
    
   
    
    chain = np.zeros((dim, N))
    chain[:,0] = np.ravel(x)
    chain2 = np.copy(chain)
    
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
        
        with nogil:
            for j in range(0,dim):

                old = chainv[j,i-1]
                #new = old + samplingsigma*samplingsigma*number[j]
                #print(new)
                new = sqrt(1.0-beta*beta)*old + beta*samplingsigma*samplingsigma*number[j]
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
                    chainv[j,i] = new
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
                    chainv[j,i] = old
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
    return  chain
