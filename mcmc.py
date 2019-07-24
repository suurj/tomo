import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
#from autograd import grad
import scipy
from line_profiler import LineProfiler
import  time
import matplotlib.pyplot as plt
#from cyt import hmc,tv_grad,cauchy_grad,tikhonov_grad,mwg_tv,mwg_cauchy

# f = 2*np.random.randn(1000,1)
# m = np.zeros((1000,))
# v = np.zeros((1000,1))
# m[0] = f[0]
# a = m[0]
# c = 0
#
# for i in range(1,1000):
#     au = 1 / (i+1) * ((i) * a + f[i])
#     m[i] = au
#     c = (i - 1) / (i) * v[i-1]*v[i-1] + 1 / i * (f[i] - a) ** 2;
#     a = au;
#     v[i] = 2.4*np.sqrt(c) + 10**-12;
#
# exit(1)

#M = sp.eye(2,format='csc')

# M = np.array([[10, 0.9],[0.9,1]])
# M = np.linalg.inv(M)
# M = np.linalg.cholesky(M)
# M = sp.csc_matrix(M)
# Lx = sp.csc_matrix((2,2),dtype='double')
# Ly = sp.csc_matrix((2,2),dtype='double')
# y = np.array([[10.0],[-17.0]],dtype='double')
# y = M.dot(y)
# x0 = np.array([[2,3]],dtype='double').T
# N = 100000
# g = mwg_cauchy(M, Lx, Ly,  y, x0,N, regalpha=1.0, samplebeta=0.3, sampsigma=0.001,lhsigma=1.0)
# g = g[:,0:100]
# print(np.mean(g,axis=1))
# print(np.cov(g))
# plt.plot(g[0,:],g[1,:],'*r')
# plt.show()

#exit(1)

from numba import jit
#@jit(nopython=True)
#@profile

'''
def hmc(N,x):
    dim = x.shape[0]
    q = np.zeros((dim, 1))
    chain = np.zeros((dim, N))
    chain[:, 0] = np.ravel(q)
    epsilon = 0.015
    L = 10
    pot_gradient = grad(potential)
    acc = 0
    numer = np.zeros((dim, 1))

    for i in range(1,N):
        p = np.random.randn(dim, 1)
        pold = p
        qold = q

        p = p - epsilon * pot_gradient(q)
        for j in range(0,L):
            q = q+epsilon*p
            if (j < L-1):
                p = p-epsilon*pot_gradient(q)

        p = p-epsilon*pot_gradient(q)/2

        proposal = np.exp(potential(qold) - potential(q) + np.sum(np.power(pold,2)) - np.sum(np.power(p,2)))
        if(np.random.rand(1)  <= proposal):
            acc = acc+1
        else:
            q = qold

        chain[:,i] = np.ravel(q)
        numer = numer + q
    print(acc/N)
    print(numer / N)
    return chain
'''
#R = np.eye(3)*1

#R = R + np.array([[0, -2, -2] ,[-2, 0 ,-2],[ -2, -2 ,0]]);
#R = np.linalg.cholesky(np.linalg.inv(R));
# def  gradient(f,x):
#     N = x.shape[0]
#     eps = 1e-7;
#     xn = x.copy(); xe = x.copy();
#     gr = np.zeros((N,1));
#     for i  in range(N):
#         xn[i] = x[i] + eps;
#         xe[i] = x[i] - eps;
#         gr[i] = (f(xn) -f(xe))/(2*eps);
#         xn[i] = x[i];
#         xe[i] = x[i];
#
#     return  gr
#
# def density(x,M,Lx,Ly,y,s2,alfa,beta):
#     Mxy = M.dot(x) - y
#     # Lxx = np.dot(Lx,x)
#     Lxx = Lx.dot(x)
#     # Lyx = np.dot(Ly,x)
#     Lyx = Ly.dot(x)
#     return -0.5/s2 * np.dot(Mxy.T, Mxy)  -alfa*np.sum(np.sqrt(np.power(Lxx,2) + beta))  -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta))
#     #return np.log(np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
#     #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta))))
#
# def density2(x,M,Lx,Ly,y,s2,alfa,beta):
#     Mxy = M.dot(x) - y
#     # Lxx = np.dot(Lx,x)
#     Lxx = Lx.dot(x)
#     # Lyx = np.dot(Ly,x)
#     Lyx = Ly.dot(x)
#     return -0.5/s2 * np.dot(Mxy.T, Mxy)  - np.sum(np.log(alfa+np.power(Lyx,2))) - np.sum(np.log(alfa+np.power(Lxx,2)))
#
# def density3(x,M,Lx,Ly,y,s2,alfa,beta):
#     Mxy = M.dot(x) - y
#     # Lxx = np.dot(Lx,x)
#     Lxx = Lx.dot(x)
#     # Lyx = np.dot(Ly,x)
#     Lyx = Ly.dot(x)
#     return -0.5/s2 * np.dot(Mxy.T, Mxy)  - alfa*np.dot(Lxx.T,Lxx) - alfa*np.dot(Lyx.T,Lyx)
#
# x = np.random.randn(40,1)
# M = np.random.randn(40,40)
# Lx = np.random.randn(43,40)
# Ly = np.random.randn(40,40)
# y = np.random.randn(40,1)
# M = csc_matrix(M)
# Lx = csc_matrix(Lx)
# Ly = csc_matrix(Ly)
# j=M.dot(x)
# j = j-y
# s2 = 1
# a = 1
# b = 1
# t = time.time()
# gr = tv_grad(x,M, Lx, Ly, y, s2, a, b)
# print(time.time() -t)
# print(gr)
# wrapper = lambda x: density(x,M,Lx,Ly,y,s2,a,b)
# t = time.time()
# gg = gradient(wrapper,x)
# print(time.time() -t)
# print(gg)
# exit(1)

# def gradi(x,Q):
#     return -0.5*2*((Q[0].T).dot(Q[0])).dot(x)
#     #*np.exp(-0.5*np.dot(res.T,res))
#     #return np.reshape(-np.matmul(np.matmul(A[0].T,A[0]),x),(-1,1))
#
#
# def logdensity(theta,Q):
#     #print(A[0].dot(theta))
#     #print(-1* np.dot(A[0].dot(theta).T, A[0].dot(theta)),theta)
#     return -0.5* np.dot(Q[0].dot(theta).T, Q[0].dot(theta))
#     #return 2 * np.dot((np.dot(A[0], theta)).T , (np.dot(A[0], theta)))
#     #return np.exp(-0.5*(np.dot(A,theta)).T@(np.dot(A,theta)) - 0.5 * np.dot(r.T, r))
#     #return np.exp(np.log(pdf(theta)) - 0.5 * np.dot(r.T, r))
#     #return np.exp(np.log(pdf(theta))-0.5*np.dot(r.T,r))

'''
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

#@profile
def buildtree(theta,r,u,v,j,epsilon,theta0,r0,Q,initialdr,currgrad):
    if (j==0):
        thetatilde, rtilde, currdensitytilde,currgradtilde = leapfrog(theta,r,v*epsilon,Q,currgrad)
        #ldensitytilde = Q.logdensity(thetatilde,Q)
        #ldensity = Q.logdensity(theta0,Q)
        logu = np.log(u)
        #print(currdensity)
        #print(initialdr)
        diff = np.min(np.array([np.exp(currdensitytilde - 0.5*np.dot(rtilde.T,rtilde) -initialdr),1]))
        joint0 = initialdr
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

def hmc(M,theta0,Q,Madapt,de=0.6,cm=False):
    np.random.seed(1)
    theta0 = np.reshape(theta0,(-1,1))
    dim =theta0.shape[0]
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

    for i in range(1,M):

        time.sleep(0.001)
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

    #print (epsilon)
    if(cm == False):
        return theta[:,Madapt:]
    else:
        return cmestimate
#print(grad(pdf)(np.array([-0.6490,1.1812,-0.7585]) ))
'''
from cyt import hmc, tfun_cauchy, tfun_tikhonov, tikhonov_grad,tfun_tv, tv_grad ,cauchy_grad,argumentspack
from pytwalk import pytwalk

def  gradient(f,Q,x):
    N = x.shape[0]
    eps = 1e-7;
    xn = x.copy(); xe = x.copy();
    gr = np.zeros((N,1));
    for i  in range(N):
        xn[i] = x[i] + eps;
        xe[i] = x[i] - eps;
        gr[i] = (f(xn,Q) -f(xe,Q))/(2*eps);
        xn[i] = x[i];
        xe[i] = x[i];

    return  gr

Q = argumentspack()
Q.s2 = 1
Q.a = 10
Q.M = np.array([[3,-1],[-1,3]])
Q.M = np.linalg.inv(Q.M)
Q.M = np.linalg.cholesky(Q.M)
Q.M = Q.M.T
Q.M = csc_matrix(Q.M)
Q.y = np.zeros((2,1))
Q.gradi = tikhonov_grad
Q.logdensity = tfun_tikhonov
#Q.logdensity = lambda x,Q: -0.5/Q.s2* np.dot(Q.M.dot(x).T, Q.M.dot(x))
#Q.gradi = lambda x,Q:  -0.5/Q.s2*2*((Q.M.T).dot(Q.M)).dot(x)
Lx=np.zeros((2,2))
#Ly=np.random.rand(3,3)
Ly = np.zeros((2,2))
Q.Lx = csc_matrix(Lx)
Q.Ly = csc_matrix(Ly)
x = np.zeros((2,))
uf = lambda x: -tfun_tikhonov(x,Q)
su = lambda x: True
t = time.time()
#ww=pytwalk(U=uf,Supp=su,n=2)
#res=ww.Run( T=50000, x0=0*np.ones(2), xp0=1*np.ones(2))
#rr = res.T
#rr = rr[0:2,:]
#ww.Hist()
rr=hmc(10000,x,Q,100,cm=False)
#print(rr)
print(time.time()-t)
print(np.cov(rr))
plt.plot(rr[0,:],rr[1,:],'*r')
plt.show()
exit(0)
#Q.logdensity = lambda x: -0.5/Q.s2*x*x
#Q.gradi = lambda x: -2*x
#print(Q.gradi,id(Q.gradi),id(j),id(Q.s2))
#Q.logdensity
#Q.logdensity = lambda x,Q: -0.5/Q.s2* np.dot(Q.M.dot(x).T, Q.M.dot(x))
#Q.gradi = lambda x,Q:  -0.5/Q.s2*2*((Q.M.T).dot(Q.M)).dot(x)
#Q.logdensity = lambda x,Q: np.log(np.exp(-(x[0]-1)/2)/(2*(1+np.exp((-x[0]-1)/2))**2)*np.exp(-(x[1]-1)/2)/(2*(1+np.exp((-x[1]-1)/2))**2))

#Q.gradi = lambda x,Q:  gradient(Q.logdensity,Q,x)

#exit(0)
#M = np.random.randn(5,3)
#M=np.zeros((3,3))
#M=csc_matrix(M)
#Lx=np.random.randn(3,3)


'''
def  gradient(f,x):
    N = x.shape[0]
    eps = 1e-7;
    xn = x.copy(); xe = x.copy();
    gr = np.zeros((N,1));
    for i  in range(N):
        xn[i] = x[i] + eps;
        xe[i] = x[i] - eps;
        gr[i] = (f(xn) -f(xe))/(2*eps);
        xn[i] = x[i];
        xe[i] = x[i];

    return  gr

def density(x):
    Mxy = M.dot(x) - y
    # Lxx = np.dot(Lx,x)
    Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Ly.dot(x)
    return -s2 * np.dot(Mxy.T, Mxy)  -alfa*np.sum(np.sqrt(np.power(Lxx,2) + beta))  -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta))
    #return np.log(np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta))))

def density2(x):
    Mxy = M.dot(x) - y
    # Lxx = np.dot(Lx,x)
    Lxx = Lx.dot(x)
    # Lyx = np.dot(Ly,x)
    Lyx = Ly.dot(x)
    return -s2 * np.dot(Mxy.T, Mxy)  - np.sum(np.log(alfa+np.power(Lyx,2))) - np.sum(np.log(alfa+np.power(Lxx,2)))


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

    common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -s2 * 2.0 * (M.T).dot(Mxy) * common


    #exit(1)
    for i in range(col):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - Lxdata[j]*Lxx[Lxindices[j]]/np.sqrt(Lxx[Lxindices[j]]**2 + beta)

        #exit(0)
        gr[i,0] = gr[i,0] + alfa*s*common

    for i in range(col):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - Lydata[j] * Lyx[Lyindices[j]] / np.sqrt(Lyx[Lyindices[j]] ** 2 + beta)

        # exit(0)
        gr[i, 0] = gr[i, 0] + alfa * s * common
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

    common = 1#np.exp(-s2 * np.dot(Mxy.T, Mxy)) * np.exp(-alfa * np.sum(np.sqrt(np.power(Lxx,2) + beta))) * np.exp(
    #   -alfa * np.sum(np.sqrt(np.power(Lyx,2) + beta)))
    gr = -s2 * 2.0 * (M.T).dot(Mxy) * common


    #exit(1)
    for i in range(col):
        s = 0
        for j in range(Lxptr[i],Lxptr[i+1]):
            s = s - 2*Lxx[Lxindices[j]]/(alfa+Lxx[Lxindices[j]]**2)*Lxdata[j]

        #exit(0)
        gr[i,0] = gr[i,0] + s*common

    for i in range(col):
        s = 0
        for j in range(Lyptr[i], Lyptr[i + 1]):
            s = s - 2 * Lyx[Lyindices[j]] / (alfa + Lyx[Lyindices[j]] ** 2)*Lydata[j]

        # exit(0)
        gr[i, 0] = gr[i, 0] +   s * common
    return gr


def gradii(x):
    #*np.exp(-0.5*np.dot(res.T,res))
    return -s2*2*((M.T).dot(M)).dot(x)
    #return np.reshape(-np.matmul(np.matmul(M.T,M),x),(-1,1))


def logdensityy(theta):
    return  -s2*np.dot(M.dot(theta).T , M.dot(theta))
'''

#j = (M.T).dot(M).dot(x)
#t = time.time()
#print(gradient(logdensity,x))
#g = (gradi(x))

#g=cauchy_grad(M,Lx,Ly,y,x,s2,alfa,beta)
#print( g)
#r=hmc(30000,np.array([1.0,0.0,0.0]))
#print(time.time()-t)
#print(np.cov(r))
#plt.plot(r[0,:],r[1,:],'*r')
#plt.show()

#y=np.random.randn(500000,1)
#R = np.random.randn(5000,5000)
#R2 = np.random.randn(5000,5000)
# R = sp.eye(500000,format='csc')
# R[1,1] = 2
# t = time.time()
#
# #f = scipy.linalg.blas.sgemm(1.0,R.T,R)
# # print(t -time.time())
# # t = time.time()
# ff = R.T.dot(R).dot(y)
# print(t -time.time())
# print(np.sum(ff))

#print(gradi(np.array([-0.6490,1.1812,-0.7585])))
# M = sp.eye(700000,format="csc") #+ np.array([[0,4.9,0],[4.9,0,0],[0,0,0]])
# #M = sp.linalg.inv(M)
# #R = sp.linalg.cholesky(M)
# #R = R.T
# R = M
#Q = np.eye(3)

'''
def koe(N,y):
    np.random.seed(1)
    dim = y.shape[0]
    x = np.zeros((dim,1))
    #C = 5*np.eye(dim)
    C = 1
    alpha = 1
    #Ci = np.linalg.inv(C)
    Ci = 1/C
    #w=pdf(-4*np.ones((30,1)))
    sigma = 1
    beta = 0.3
    chain = np.zeros((dim, N))
    chain2 = np.zeros((dim, N))
    chain[:,0] = np.ravel(x)
    chain2[:,0] = np.ravel(x)
    acc = 0
    xc = np.copy(x)

    #numer = 0
    for i in range(1,N):
        randoms = np.random.randn(dim, )
        accept = np.random.rand(dim, )

        for j in range(0,dim):

            pr = np.sqrt(1-beta**2.0)*x[j,0] + beta*sigma*sigma*randoms[j]
            #pr = x[j,0] + sigma*randoms[j]
            #print(pr)
            # rg = M[:,j]
            # change = np.reshape(M[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            # change2 = np.reshape(L[:,j]*(np.ravel(pr-x[j,0])),(-1,1))
            # #newvalue=np.exp((change-y+w).T@(change-y+w)*Ci*-1/2)
            # newvalue = np.ravel(np.exp((change - y + w).T @ (change - y + w) * -Ci))
            # newvalue2 = np.exp(-alpha*np.sum(np.abs(change2+w2)))
            # ratio2 = newvalue*newvalue2/(value*value2)
            xc[j,0] = pr
            ratio = pdf(xc)/pdf(x)
            chain2[j, i] = ratio
            if (accept[j] <= ratio):
                x[j,0] = pr#xc[j,0]
                # w = w+ change
                # w2 = w2+change2
                # value = newvalue
                #e = pdf(x)
                # value2 = newvalue2
                #chain[j,i] = pr
                acc = acc +1
            else:
                #chain[j, i] = x[j,0]
                #pass
                xc[j,0] = x[j,0]

        #numer = numer + x
        chain[:,i] = np.ravel(x)
    #print(acc/(N*dim))
    #print(numer/N)
    return  chain

def am(N,x):
    print(pdf(x))
    dim = x.shape[0]
    C = np.eye(dim)
    R = np.linalg.cholesky(C)
    chain = np.zeros((dim,N))
    chain[:,0] = np.ravel(x)
    xmeanpr = x
    diag = np.eye(dim)
    epsilon = 0.01
    sd = 2.4**2.0/dim
    acc = 0
    numer = np.zeros((dim,1))
    density = pdf(x)
    for i in range(1,N):
        xn = x +  R@np.random.randn(dim,1)
        densitynew =  pdf(xn)
        ratio =densitynew/density
        if(np.random.rand(1)  <= ratio):
            x = xn
            density = densitynew
            acc = acc + 1

        chain[:,i] = np.ravel(x)
        numer = numer + x
        n = i + 1
        xmean = 1 / (n) * ((n - 1) * xmeanpr + x)
        if(i>=1):
            #C = sd*np.cov(chain[:,0:i+1]) + epsilon*diag
            C = (n-1)/n*C + sd/n*(n*xmeanpr@xmeanpr.T - (n+1)*xmean@xmean.T + x@x.T + epsilon*diag)
            R = np.linalg.cholesky(C)

        xmeanpr = xmean
        #print(C)
    print(acc/N)
    #print(numer/N)
    return chain
'''
# qq = 10 * np.eye(30)
# qqi = np.linalg.inv(qq)
#
# def pdff(x):
#     return np.ravel(np.exp(-1.0 / 2.0 * x.T @ qqi @ x))
#
# def pdf2(x):
#     return (np.abs(x[0])<=2)*(np.abs(x[1])<=2)*np.exp((-10*(x[0]**2.0-x[1])**2.0-(x[1]-1/4)**4.0))

# def ex(x,y):
#     return x*(np.abs(x)<=2)*(np.abs(y)<=2)*np.exp((-10.0*(x**2.0-y)**2.0-(y-1/4)**4.0))/1.1813446034359008
# def ey(x,y):
#     return y*(np.abs(x)<=2)*(np.abs(y)<=2)*np.exp((-10.0*(x**2.0-y)**2.0-(y-1/4)**4.0))/1.1813446034359008


# def potential(x):
#     return -np.log((np.abs(x[0])<=2)*(np.abs(x[1])<=2)*np.exp((-10*(x[0]**2.0-x[1])**2.0-(x[1]-1/4)**4.0)))

# v = integrate.dblquad(ex, -2, 2, lambda x: -2, lambda x: 2)
# w = integrate.dblquad(ey, -2, 2, lambda x: -2, lambda x: 2)
# print([v[0],w[0]])

#w = am(500,x)
# print(t-time.time())
# t = time.time()
#
# Lx = sp.eye(1,700000)
# Ly = sp.eye(1,700000)
# y = 4*np.zeros((700000,1))
# x0 = np.zeros((700000,1))
# print(np.sum(np.power(R@y,2)))
# print(np.sum(y.T@M@y))
# #c = am(5,x0)
# #c = koe(1009,y)
# print("----")
# c = mwg(R,Lx,Ly,y,x0,20 ,regalpha=1, samplebeta=0.1, sampsigma=5.0,lhsigma=1.0)
# #print(c)
# #qq = w-c
# print(t-time.time())
# plt.plot(c[0,:],c[1,:],'r*')
# #print(np.cov(c))
# #print(np.mean(c,axis=1))
# #plt.plot(w[0,:],w[1,:],'b*')
# plt.show()