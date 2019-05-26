import autograd_sparse as sp
from autograd import elementwise_grad as egrad
import os
import autograd.numpy as np
from autograd import grad, jacobian
from scipy.linalg import circulant
from skimage.transform import radon, rescale
from skimage.io import imread
from numba import jit


@jit(nopython=True)
def matrix(image, N_theta, N_r, dim):

    theta = np.linspace(0., 180., N_theta, endpoint=False)
    flattened = np.reshape(image, (-1, 1))
    g = "G"
    fname = "radonmatrix/" + str(N_r) + "x" + str(N_theta) + ".npz"

    if (not os.path.isfile(fname)):
        M = np.zeros([N_r * N_theta, dim * dim])
        empty = np.zeros([dim, dim])
        for i in range(0, dim):
            for j in range(0, dim):
                empty[i, j] = 1
                M[:, i * dim + j] = np.ravel(
                    np.reshape(radon(empty, theta, circle=True), (N_r * N_theta, 1)))
                empty[i, j] = 0
        np.savez_compressed(fname, radonoperator=M)

scale = 0.2
image = imread("shepp.png", as_gray=True)
image = rescale(image, scale=scale, mode='edge', multichannel=False)
N_theta = 50
theta = np.linspace(0., 180., N_theta, endpoint=False)
(N_r, N_theta) = (radon(image, theta, circle=True)).shape
(dim, dimx) = image.shape
matrix(image,N_theta,N_r,dim)
# d = 11
# x0= 1+ 0.05*np.random.randn(11 , 1)
# regoperator = circulant(np.block([[2], [-1] , [np.zeros((11 - 3, 1))], [-1]]))
# #regoperator = np.kron(np.eye(11), regoperator) + np.kron(regoperator, np.eye(11))
# radonoperator = np.random.randn(13,11)
# radonoperator = sp.csr_matrix(radonoperator)
# regoperator = sp.csr_matrix(regoperator)
# alpha = 10
# x1 = 1+ 0.05*np.random.randn(13, 1)
# 
# 
# def mrf( x):
#     x = np.reshape(x, (-1, 1))
#     a = np.sum(np.power(sp.dot(radonoperator, x) - x1, 2))
#     b = np.sum(np.power(sp.dot(regoperator, x), 2))
#     return  np.sum(np.array([a,b]))
# 
# 
# def agrad(x):
#     gradient_handle = grad(mrf)
#     return (gradient_handle(x))
# 
# 
# e = agrad(x0)
# print(e)
# solution = minimize(mrf,x0,method='Newton-CG',jac=agrad,options={'disp': True})
#



# N = 5
#
# # ----- pytest fixture for sparse arrays ----- #
#
# def eye():
#     return sp.eye(N).tocsr()
#
#
# def sp_rand():
#     return sp.sp_rand(N).tocsr()
#
#
#
# def test_sparse_coo_matrix():
#     """This just has to not error out."""
#     data = np.array([1, 2, 3]).astype('float32')
#     rows = np.array([1, 2, 3]).astype('float32')
#     cols = np.array([1, 3, 4]).astype('float32')
#     sparse = sp.coo_matrix(data, (rows, cols))
#
#
# # ----- tests for array multiplication ----- #
#
# def test_sparse_dense_multiplication(eye):
#     """This just has to not error out."""
#     dense = np.random.random(size=(N, N))
#     sp.dot(eye, dense)
#     sp.dot(dense, eye)
#
#
#
# # ----- tests for dot product ----- #
#
#
# # differentiating with respect to sparse argument (will fail)
#
# def test_sparse_dot_0_2(eye):
#     dense = np.random.random(size=(N, N))
#     sparse = eye
#     def fun(x):
#         return sp.dot(x, dense)
#     check_grads(fun)(sparse)
#
#
# # differentiating with respect to dense argument (will pass)
# # dense.ndim = 1
#
# def test_sparse_dot_1_1(eye):
#     dense = np.random.random(size=(N, ))
#     sparse = eye
#     def fun(x):
#         return sp.dot(sparse, x)
#     check_grads(fun)(dense)
#
#
# def test_sparse_dot_1_2(sp_rand):
#     dense = np.random.random(size=(N, N))
#     sparse = sp_rand
#     def fun(x):
#         return sp.dot(sparse, x)
#     check_grads(fun)(dense)
#
#
# # ----- tests of spsolve ----- #
#
#
#
# def test_sparse_spsolve_0_2(sp_rand):
#     dense = np.random.random(size=(N, N))
#     sparse = sp_rand
#     def fun(x):
#         return sp.spsolve(x, dense)
#     check_grads(fun)(sparse)
#
#
# def test_sparse_spsolve_1_2(sp_rand):
#     dense = np.random.random(size=(N, N))
#     sparse = sp_rand
#     def fun(x):
#         return sp.spsolve(sparse, x)
#     check_grads(fun)(dense)
#
#
#
# q = test_sparse_dot_1_1(eye())
# w = test_sparse_spsolve_0_2(eye())