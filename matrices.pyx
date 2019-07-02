import numpy as np
cimport numpy as np
cimport cython
import scipy.sparse as sp
from cython.parallel import prange
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import math
from libc.math cimport sqrt,fabs,exp, cos, tan,sin,M_SQRT2,M_PI,abs
from libcpp.list cimport list as cpplist
from libcpp.deque cimport deque
from libcpp.vector cimport vector
import time
from cython.parallel import threadid as thid
from libc.stdlib cimport malloc 

def totalmatrix(n,levels,g,h):
    if (levels<1 or np.mod(n,2**levels) != 0 ):
        raise Exception('DWT level mismatch.')

    Gs = []
    Hs = []
    Hprev = []
    for i in range(0,levels):
        Gi, Hi = waveletonce(g, h, int(n/(2**(i))))
        Gs.append(Gi)
        Hs.append(Hi)
        if (len(Hprev) == 0):
            Hprev.append(Hi)
        else:
            Hprev.append(Hi.dot(Hprev[i-1]))

    for i in range(0,levels-1):
        if (i == 0):
            M = sp.kron(Gs[i],Gs[i])
            M = sp.vstack((sp.kron(Gs[0],Hs[0]),M))
            M = sp.vstack((sp.kron(Hs[0], Gs[0]), M))
        else:
            p = Hprev[i-1]
            gp = Gs[i].dot(p)
            hp = Hs[i].dot(p)
            M = sp.vstack((sp.kron(gp,gp),M))
            M = sp.vstack((sp.kron(gp, hp), M))
            M = sp.vstack((sp.kron(hp, gp), M))

    if (levels ==1):
        gp = Gs[0]
        hp = Hs[0]
        M = sp.kron(gp, gp)
        M = sp.vstack((sp.kron(gp, hp), M))
        M = sp.vstack((sp.kron(hp, gp), M))
        M = sp.vstack((sp.kron(hp, hp), M))
    else:
        p = Hprev[levels - 2]
        gp = Gs[levels-1].dot(p)
        hp = Hs[levels-1].dot(p)
        M = sp.vstack((sp.kron(gp, gp), M))
        M = sp.vstack((sp.kron(gp, hp), M))
        M = sp.vstack((sp.kron(hp, gp), M))
        M = sp.vstack((sp.kron(hp, hp), M))

    return  csc_matrix(M)

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
    cdef vector[int] Wrow
    cdef vector[int] Wcol
    cdef vector[double] Gdata
    cdef vector[double] Hdata
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
        coo_rowv[i] = Wrow.back()
        Wrow.pop_back()
        
        coo_colv[i] = Wcol.back()
        Wcol.pop_back()

        Hcoo_datav[i] = Hdata.back()
        Hdata.pop_back()
        Gcoo_datav[i] = Gdata.back()
        Gdata.pop_back()
    
    
    G=coo_matrix((Gcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #GG = csc_matrix(GG)
    H=coo_matrix((Hcoo_data, (coo_row, coo_col)), shape=(nhalf,n))
    #HH = csc_matrix(HH)
    #P = sp.vstack([HH,GG],format='csc')
    
    return (G,H)  

ctypedef vector[int]* diptr
ctypedef vector[double]* dfptr
@cython.boundscheck(False) 
#@cython.wraparound(False)
@cython.cdivision(True) 
def  radonmatrix(size,theta,Nthreads=4):
    #cdef vector[int] row
    #cdef vector[int] col
    #cdef vector[double] data
    
    cdef int Nth = Nthreads
    cdef int i
    
    mrows = new vector[diptr](Nth)
    mcols = new vector[diptr](Nth)
    mdata = new vector[dfptr](Nth)
    row = mrows[0]
    col = mcols[0]
    data = mdata[0]
    
    for i in range(Nth):
        row[i] = new vector[int]()
        col[i] = new vector[int]()
        data[i] = new vector[double]()
    
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
    cdef int t,m,n,r,th
    if (T == 1):
        dt = 0
    else:
        dt = theta[1]-theta[0]
        
    
    start = time.time() 
    with nogil:                     
        for r in prange (0,R,num_threads=Nth):
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
                            th = thid()
                            row[th].push_back(r*T+t)
                            col[th].push_back(n*M+m)
                            data[th].push_back(ray)
                    #A(r*T + t+1, n*M+m+1) = 

            
    print(time.time()-start)     
     
                     

    #push_back = temp.push_back
    #for x in range(5):
    #    row.push_back(x)
    #    col.push_back(x+1)
    #    data.push_back(2.0)
    cdef int Nel = 0
    for i in range(0,Nth):
        Nel = Nel + row[i].size()

    #cdef int Nel = row.size()
    coo_row  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_rowv  = coo_row
    coo_col  = np.zeros((Nel,),dtype=np.int32)
    cdef int [:] coo_colv  = coo_col
    coo_data = np.zeros((Nel,))
    cdef double [:] coo_datav  = coo_data

    #front = temp.front()
    #pop_front = temp.pop_front()
    i = 0
    for j in range(0,Nth):
        for k in range(row[j].size()):
            coo_rowv[i] = row[j].back()
            row[j].pop_back()

            coo_colv[i] = col[j].back()
            col[j].pop_back()

            coo_datav[i] = data[j].back()
            data[j].pop_back()
            i = i + 1

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