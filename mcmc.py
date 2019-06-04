from skimage.io import imread
import scipy.sparse as sp
from skimage.transform import radon, rescale
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
from autograd import grad
import scipy
from scipy.linalg import circulant
from scipy.optimize import  minimize
from line_profiler import LineProfiler
import  time
import matplotlib.pyplot as plt
from cyt import mwg

from numba import jit
#@jit(nopython=True)
#@profile


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


M = sp.eye(700000,format="csc") #+ np.array([[0,4.9,0],[4.9,0,0],[0,0,0]])
#M = sp.linalg.inv(M)
#R = sp.linalg.cholesky(M)
#R = R.T
R = M
#Q = np.eye(3)

def pdf(x):
    #return np.exp(-1/2*x.T@Q@x)
    #return np.exp(-1/2*np.sum(np.power(R@x,2)))
    return np.exp(-1/2*(R@x).T@(R@x))
    #return np.exp(-1/2*x.T@M@x)

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

qq = 10 * np.eye(30)
qqi = np.linalg.inv(qq)

def pdff(x):
    return np.ravel(np.exp(-1.0 / 2.0 * x.T @ qqi @ x))

def pdf2(x):
    return (np.abs(x[0])<=2)*(np.abs(x[1])<=2)*np.exp((-10*(x[0]**2.0-x[1])**2.0-(x[1]-1/4)**4.0))

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
t = time.time()

Lx = sp.eye(1,700000)
Ly = sp.eye(1,700000)
y = 4*np.zeros((700000,1))
x0 = np.zeros((700000,1))
print(np.sum(np.power(R@y,2)))
print(np.sum(y.T@M@y))
#c = am(5,x0)
#c = koe(1009,y)
print("----")
c = mwg(R,Lx,Ly,y,x0,20 ,regalpha=1, samplebeta=0.1, sampsigma=5.0,lhsigma=1.0)
#print(c)
#qq = w-c
print(t-time.time())
plt.plot(c[0,:],c[1,:],'r*')
#print(np.cov(c))
#print(np.mean(c,axis=1))
#plt.plot(w[0,:],w[1,:],'b*')
plt.show()