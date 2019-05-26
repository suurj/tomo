from skimage.io import imread
import autograd_sparse as sp
from skimage.transform import radon, rescale
import autograd.numpy as np
import warnings
from autograd import grad
from scipy.linalg import circulant
from scipy.optimize import  minimize
from scipy import integrate
import matplotlib.pyplot as plt

#from numba import jit
#@jit(nopython=False)
def hmc(N,dim):
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

def am(N,dim):

    C = np.eye(dim)
    R = np.linalg.cholesky(C)
    x = np.zeros((dim,1))
    chain = np.zeros((dim,N))
    chain[:,0] = np.ravel(x)
    xmeanpr = x
    diag = np.eye(dim)
    epsilon = 0.01
    sd = 2.4**2.0/dim
    acc = 0
    numer = np.zeros((dim,1))
    density = pdf2(x)
    for i in range(1,N):
        xn = x +  R@np.random.randn(dim,1)
        densitynew =  pdf2(xn)
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
    print(numer/N)
    return chain

q =  10*np.eye(30)
r = np.linalg.cholesky(q)
qi = np.linalg.inv(q)


def pdf(x):
    return np.ravel(np.exp(-1.0/2.0*x.T@qi@x))

def pdf2(x):
    return (np.abs(x[0])<=2)*(np.abs(x[1])<=2)*np.exp((-10*(x[0]**2.0-x[1])**2.0-(x[1]-1/4)**4.0))

# def ex(x,y):
#     return x*(np.abs(x)<=2)*(np.abs(y)<=2)*np.exp((-10.0*(x**2.0-y)**2.0-(y-1/4)**4.0))/1.1813446034359008
# def ey(x,y):
#     return y*(np.abs(x)<=2)*(np.abs(y)<=2)*np.exp((-10.0*(x**2.0-y)**2.0-(y-1/4)**4.0))/1.1813446034359008


def potential(x):
    return -np.log((np.abs(x[0])<=2)*(np.abs(x[1])<=2)*np.exp((-10*(x[0]**2.0-x[1])**2.0-(x[1]-1/4)**4.0)))

# v = integrate.dblquad(ex, -2, 2, lambda x: -2, lambda x: 2)
# w = integrate.dblquad(ey, -2, 2, lambda x: -2, lambda x: 2)
# print([v[0],w[0]])
w = am(500,2)
c = hmc(500,2)
plt.plot(c[0,:],c[1,:],'r*')
plt.plot(w[0,:],w[1,:],'b*')
plt.show()