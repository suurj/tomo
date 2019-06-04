from skimage.io import imread
import autograd_sparse as sp
from skimage.transform import radon, rescale
import autograd.numpy as np
import warnings
from autograd import grad
from scipy.linalg import circulant
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix
from scipy.optimize import  minimize
import time
import os
import matplotlib.pyplot as plt



class tomography:

    def __init__(self, filename,scaling=0.1,ntheta=40,noise=0.000000):
        self.image = imread(filename, as_gray=True)
        self.image = rescale(self.image, scale=scaling, mode='edge', multichannel=False)
        (self.dim, self.dimx) = self.image.shape
        if (self.dim != self.dimx):
            raise Exception('Image is not rectangular.')
        self.theta = np.linspace(0., 180., ntheta, endpoint=False)
        self.flattened = np.reshape(self.image, (-1, 1))
        (self.N_r, self.N_theta) = (self.radonww(self.image, self.theta, circle=True)).shape
        fname = 'radonmatrix/'+'0_180-' + str(self.N_r) + 'x' + str(self.N_theta) + '.npz'

        if (not os.path.isfile(fname)):
            # Mf = np.zeros([N_r * N_theta, dim * dim])
            M = lil_matrix((self.N_r * self.N_theta, self.dim * self.dim))
            empty = np.zeros([self.dim, self.dim])
            for i in range(0, self.dim):
                for j in range(0, self.dim):
                    empty[i, j] = 1
                    ww = np.ravel(np.reshape(radon(empty, self.theta, circle=True), (self.N_r * self.N_theta, 1)))
                    M[:, i * self.dim + j] = np.reshape(radon(empty, self.theta, circle=True), (self.N_r * self.N_theta, 1))
                    empty[i, j] = 0
            sp.save_npz(fname,M)

        self.radonoperator =sp.load_npz(fname)
        #self.radonoperator = loaded['radonoperator']
        #loaded.close()

        self.measurement = np.exp(-self.radonoperator @ self.flattened) + noise * np.random.randn(self.N_r * self.N_theta, 1)
        self.lines = -np.log(self.measurement)

    def map_tikhonov(self,alpha=1.0):
        #col = np.block([[-1], [np.zeros((self.y - 2, 1))]])
        #row = np.block([np.array([-1,2,-1]),np.zeros((self.x-4,))])
        d1= circulant(np.block([[2], [-1] , [np.zeros((self.dim - 3, 1))], [-1]]))
        self.regx = np.kron(np.eye(self.dim), d1)
        self.regy =  np.kron(d1, np.eye(self.dim))
        self.alpha = alpha
        self.radonoperator = sp.csr_matrix(self.radonoperator)
        self.regx = sp.csr_matrix(self.regx)
        self.regy = sp.csr_matrix(self.regy)

        # import scipy
        # b = np.block([[self.lines], [np.zeros((self.dim * self.dim, 1))]])
        # A = np.block([[self.radonoperator],[ np.sqrt(self.alpha)*self.regoperator]])
        # A = scipy.sparse.csr_matrix(A)
        # solution = scipy.sparse.linalg.lsqr(A, b)
        # solution = np.reshape(solution[0],(self.dim,self.dim))

        #

        x0= 1+ 0.05*np.random.randn(self.dim * self.dim, 1)
        solution = minimize(self.tfun_tikhonov,x0,method='Newton-CG',jac=self.grad_tikhonov,options={ 'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return  solution


    def tfun_tikhonov(self,x):
        x = np.reshape(x, (-1, 1))
        qq = sp.dot(self.radonoperator, x)
        a = np.sum(np.power(qq - self.lines, 2))
        b1 = np.multiply(np.sum(np.power(sp.dot(self.regx, x), 2)),self.alpha)
        b2 =  np.multiply(np.sum(np.power(sp.dot(self.regy, x), 2)),self.alpha)
        return np.sum(np.array([a,b1,b2]))

    def grad_tikhonov(self,x):
        gradient_handle = grad(self.tfun_tikhonov)
        return(gradient_handle(x))

    def map_tv(self,alpha=1.0):
        reg1d= circulant(np.block([[-1], [0], [np.zeros((self.dim - 3, 1))], [1]]))
        self.regx = np.kron(np.eye(self.dim), reg1d)
        self.regy = np.kron(reg1d,np.eye(self.dim))
        self.radonoperator = sp.csr_matrix(self.radonoperator)
        self.regx = sp.csr_matrix(self.regx)
        self.regy = sp.csr_matrix(self.regy)
        self.alpha = alpha

        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, 1)
        solution = minimize(self.tfun_tv, x0, method='Newton-CG', jac=self.grad_tv, options={'maxiter':50,'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return solution

    def tfun_tv(self, x):
        beta = 0.5
        x = np.reshape(x, (-1, 1))
        qq = sp.dot(self.radonoperator, x)
        a = np.sum(np.power(qq - self.lines, 2))
        #b1 = np.sum(np.sqrt(np.sum(np.array([np.power(sp.dot(self.regx, x),2),beta]) ) ))
        b1 = np.sum(np.sqrt(np.power(sp.dot(self.regx, x),2)+beta))
        b1 = np.multiply(b1, self.alpha)
        #b2 = np.sum(np.sqrt(np.sum(np.array([np.power(sp.dot(self.regy, x), 2), beta]))))7
        b2 = np.sum(np.sqrt(np.power(sp.dot(self.regy, x), 2)+ beta))
        b2 = np.multiply(b2, self.alpha)
        r = np.sum(np.array([a,b1,b2]))
        return r

    def grad_tv(self, x):
        gradient_handle = grad(self.tfun_tv)
        return (gradient_handle(x))

    def map_cauchy(self,alpha=1.0):
        reg1d= circulant(np.block([[-1], [0], [np.zeros((self.dim - 3, 1))], [1]]))
        self.regx = np.kron(np.eye(self.dim), reg1d)
        self.regy = np.kron(reg1d,np.eye(self.dim))
        #plt.spy(self.radonoperator)
        #print(np.sum(self.radonoperator[:,1000]))
        #plt.show()
        self.radonoperator = sp.csr_matrix(self.radonoperator)
        self.regx = sp.csr_matrix(self.regx)
        self.regy = sp.csr_matrix(self.regy)
        self.alpha = alpha
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, 1)

        solution = minimize(self.tfun_cauchy, x0, method='Newton-CG', jac=self.grad_cauchy, options={'maxiter':50,'disp': True})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        #plt.imshow(solution, cmap="gray")
        #plt.show()
        return solution

    def tfun_cauchy(self, x):
        alpha = self.alpha
        x = np.reshape(x, (-1, 1))
        qq = sp.dot(self.radonoperator, x)
        a = np.sum(np.power(qq - self.lines, 2))
        b1 = np.log(alpha+np.power(sp.dot(self.regx,x),2))
        b1 = np.sum(b1)
        b2 = np.log(alpha+np.power(sp.dot(self.regy,x),2))
        b2 = np.sum(b2)
        r = np.sum(np.array([a,b1,b2]))
        return r

    def grad_cauchy(self, x):
        gradient_handle = grad(self.tfun_cauchy)
        return (gradient_handle(x))

    def plot(self):
        #self.q = self.radonoperator @ self.flattened
        #self.q = np.reshape(self.q, (self.N_r, self.N_theta))
        # qr =  radon(image, theta,circle=True)
        # q = iradon_sart(q, theta=theta)
        plt.imshow(self.image, cmap="gray")
        plt.show()

    def radonww(self,image,theta,circle=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return radon(image,theta,circle)


if __name__ == "__main__":


    t = tomography("shepp.png",0.08,20)
    r=t.map_cauchy(1)
    #r = t.map_tikhonov(10.0)
    plt.imshow(r)
    plt.figure()
    r = t.map_tv(10.0)
    plt.imshow(r)
    plt.show()


#
        #