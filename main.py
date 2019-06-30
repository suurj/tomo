from skimage.io import imread
#import autograd_sparse as sp
from skimage.transform import radon, rescale
#import autograd.numpy as np
import warnings
import numpy as np
#from autograd import grad
#from scipy.sparse import csc_matrix,csc_matrix,lil_matrix
from scipy.optimize import  minimize
import time
import math
import scipy.sparse as sp
from cyt import  radonmatrix
import os
import matplotlib.pyplot as plt
from cyt import tv_grad,cauchy_grad,tikhonov_grad



class tomography:

    def __init__(self, filename,scaling=0.1,ntheta=40,noise=0.000000):
        self.image = imread(filename, as_gray=True)
        self.image = rescale(self.image, scale=scaling, mode='edge', multichannel=False)
        (self.dim, self.dimx) = self.image.shape
        if (self.dim != self.dimx):
            raise Exception('Image is not rectangular.')
        self.theta = np.linspace(0., 179., ntheta, endpoint=True)
        self.theta=self.theta/360*2*np.pi
        self.flattened = np.reshape(self.image, (-1, 1))
        (self.N_r, self.N_theta) = (math.ceil(np.sqrt(2)*self.dim),ntheta)#self.radonww(self.image, self.theta, circle=True)).shape
        fname = 'radonmatrix/'+ 'full-' +str(self.dim) + 'x' + str(self.N_theta) + '.npz'
        #fname = 'koe.npz'

        if (not os.path.isfile(fname)):
            # M = np.zeros([self.N_r * self.N_theta, self.dim * self.dim])
            # #M = sp.lil_matrix((self.N_r * self.N_theta, self.dim * self.dim))
            # empty = np.zeros([self.dim, self.dim])
            # for i in range(0, self.dim):
            #     for j in range(0, self.dim):
            #         empty[i, j] = 1
            #         ww = np.ravel(np.reshape(radon(empty, self.theta, circle=False), (self.N_r * self.N_theta, 1)))
            #         M[:, i * self.dim + j] = ww# np.reshape(radon(empty, self.theta, circle=False), (self.N_r * self.N_theta, 1))
            #         empty[i, j] = 0
            # M = sp.csc_matrix(M)
            # sp.save_npz(fname,M)

            self.radonoperator= radonmatrix(self.dim, self.theta)
            sp.save_npz(fname,self.radonoperator)

        self.radonoperator = sp.load_npz(fname)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        #self.radonoperator = loaded['radonoperator']
        #loaded.close()

        # self.measurement = np.exp(-self.radonoperator @ self.flattened) + noise * np.random.randn(self.N_r * self.N_theta, 1)
        # self.measurement[self.measurement<=0] = 10**(-19)
        # self.lines = -np.log(self.measurement)

        self.measurement = self.radonoperator @ self.flattened
        #self.measurement[self.measurement <= 0] = 10 ** (-19)
        max = np.max(self.measurement)
        self.lines =  self.measurement + max*noise * np.random.randn(self.N_r * self.N_theta, 1)
        self.lhsigmsq = 0.5
        self.beta = 0.01

    def map_tikhonov(self,alpha=1.0):
        #col = np.block([[-1], [np.zeros((self.y - 2, 1))]])
        #row = np.block([np.array([-1,2,-1]),np.zeros((self.x-4,))])
        # d1= circulant(np.block([[2], [-1] , [np.zeros((self.dim - 3, 1))], [-1]]))
        # self.regx = np.kron(np.eye(self.dim), d1)
        # self.regy =  np.kron(d1, np.eye(self.dim))
        regvalues = np.array([2, -1, -1, -1, -1])
        offsets = np.array([0, 1, -1, self.dim - 1, -self.dim + 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        self.regx = sp.kron(sp.eye(self.dim), reg1d)
        self.regy = sp.kron(reg1d, sp.eye(self.dim))
        self.alpha = alpha
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        self.regx = sp.csc_matrix(self.regx)
        self.regy = sp.csc_matrix(self.regy)

        # import scipy
        # b = np.block([[self.lines], [np.zeros((self.dim * self.dim, 1))]])
        # A = np.block([[self.radonoperator],[ np.sqrt(self.alpha)*self.regoperator]])
        # A = scipy.sparse.csc_matrix(A)
        # solution = scipy.sparse.linalg.lsqr(A, b)
        # solution = np.reshape(solution[0],(self.dim,self.dim))

        #

        x0= 1+ 0.05*np.random.randn(self.dim * self.dim, 1)
        solution = minimize(self.tfun_tikhonov,x0,method='L-BFGS-B',jac=self.grad_tikhonov,options={'maxiter':20, 'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return  solution


    def tfun_tikhonov(self,x):
        x = np.reshape(x, (-1, 1))
        Mxy = self.radonoperator.dot(x) - self.lines
        # Lxx = np.dot(Lx,x)
        Lxx = self.regx.dot(x)
        # Lyx = np.dot(Ly,x)
        Lyx = self.regy.dot(x)
        return 0.5/self.lhsigmsq * np.dot(Mxy.T, Mxy) + self.alpha*np.dot(Lxx.T,Lxx) + self.alpha*np.dot(Lyx.T,Lyx)
        #return np.sum(np.array([a,b1,b2]))

    def grad_tikhonov(self,x):
        x = x.reshape((-1, 1))
        # print(self.radonoperator.shape,x.shape)
        q = -tikhonov_grad(x, self.radonoperator, self.regx, self.regy, self.lines, self.lhsigmsq, self.alpha, 0.5)
        # print(np.ravel(q))
        return (np.ravel(q))

    def map_tv(self,alpha=1.0):
        # reg1d= circulant(np.block([[-1], [0], [np.zeros((self.dim - 3, 1))], [1]]))
        # self.regx = np.kron(np.eye(self.dim), reg1d)
        # self.regy = np.kron(reg1d,np.eye(self.dim))

        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        self.regx = sp.kron(sp.eye(self.dim), reg1d)
        self.regy = sp.kron(reg1d, sp.eye(self.dim))
        self.regx = sp.csc_matrix(self.regx)
        self.regy = sp.csc_matrix(self.regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        self.alpha = alpha

        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, 1)
        solution = minimize(self.tfun_tv, x0, method='L-BFGS-B', jac=self.grad_tv, options={'maxiter':230,'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return solution

    def tfun_tv(self, x):
        x = np.reshape(x, (-1, 1))
        Mxy = self.radonoperator.dot(x) - self.lines
        # Lxx = np.dot(Lx,x)
        Lxx = self.regx.dot(x)
        # Lyx = np.dot(Ly,x)
        Lyx = self.regy.dot(x)
        q =  0.5/self.lhsigmsq * np.dot(Mxy.T, Mxy) + self.alpha * np.sum(np.sqrt(np.power(Lxx, 2) + self.beta)) + self.alpha * np.sum(
            np.sqrt(np.power(Lyx, 2) + self.beta))
        return (np.ravel(q))
        #return r

    def grad_tv(self, x):
        x = x.reshape((-1, 1))
        # print(self.radonoperator.shape,x.shape)
        q = -tv_grad(x, self.radonoperator, self.regx, self.regy, self.lines, self.lhsigmsq, self.alpha, self.beta)
        #print(np.ravel(q))
        return (np.ravel(q))

    def map_cauchy(self,alpha=1.0):
        # reg1d= circulant(np.block([[-1], [0], [np.zeros((self.dim - 3, 1))], [1]]))
        # self.regx = sp.kron(sp.eye(self.dim), reg1d)
        # self.regy = sp.kron(reg1d,sp.eye(self.dim))
        #plt.spy(self.radonoperator)
        #print(np.sum(self.radonoperator[:,1000]))
        #plt.show()
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        self.regx = sp.kron(sp.eye(self.dim), reg1d)
        self.regy = sp.kron(reg1d, sp.eye(self.dim))
        self.regx = sp.csc_matrix(self.regx)
        self.regy = sp.csc_matrix(self.regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        self.alpha = alpha
        #print(self.radonoperator.shape)
        #print(self.regx.shape)
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, 1)

        solution = minimize(self.tfun_cauchy, x0, method='L-BFGS-B', jac=self.grad_cauchy, options={'maxiter':150,'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return solution

    def tfun_cauchy(self, x):
        x = np.reshape(x, (-1, 1))
        Mxy = self.radonoperator.dot(x) - self.lines
        Lxx = self.regx.dot(x)
        # Lyx = np.dot(Ly,x)
        Lyx = self.regy.dot(x)
        return   0.5/self.lhsigmsq*np.dot(Mxy.T, Mxy) + np.sum(np.log(self.alpha + np.power(Lyx, 2))) + np.sum(
            np.log(self.alpha + np.power(Lxx, 2)))
        #return r

    def grad_cauchy(self, x):
        x = x.reshape((-1,1))
        #print(self.radonoperator.shape,x.shape)
        q  =  -cauchy_grad(x,self.radonoperator, self.regx, self.regy, self.lines, self.lhsigmsq, self.alpha, 0.01)
        #print(np.ravel(q))
        return (np.ravel(q))

    def target(self):
        return self.image

    def radonww(self,image,theta,circle=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return radon(image,theta,circle)


if __name__ == "__main__":

    #np.random.seed(1)
    t = tomography("shepp128.png",1.0,15,0.05)
    #t = tomography("shepp.png",0.1,20,0.2)
    #r = t.map_tv(19.48)
    # tt = time.time()
    r=t.map_cauchy(0.01)
    # r = t.map_tikhonov(10.0)
    # print(time.time()-tt)
    #
    plt.imshow(r)
    plt.figure()
    #q = iradon_sart(q, theta=theta)
    #r = t.map_tikhonov(50.0)
    #tt = time.time()
    #r = t.target()
    r = t.map_tv(5.0)
    #print(time.time()-tt)
    plt.imshow(r)
    plt.show()


#
        #