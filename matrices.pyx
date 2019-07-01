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
    
    
    G=coo_matrix((Gcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #GG = csc_matrix(GG)
    H=coo_matrix((Hcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #HH = csc_matrix(HH)
    #P = sp.vstack([HH,GG],format='csc')
    
    return (G,H)  


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
cdef inline double gs(double p,double t) nogil:
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
            a = sqrt(4.0+(x1-x1m)*(x1-x1m))
            return a
            #disp("TOO")
            
        elif (x1 < 1.0 and x1m  > 1.0):
            a = sqrt((1.0-x1)*(1.0-x1) + (1.0-y1)*(1.0-y1))
            return a
            #disp('RRR')
            
        elif (x1 >=1.0):
            a = 0.0
            return a
            
        else:
            return -9.0