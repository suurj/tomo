import os
import numpy as np
import scipy.sparse as sp
#import autograd_sparse as sp
#import autograd.numpy as np
from sys import getsizeof
from skimage.transform import radon, rescale
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix, coo_matrix, dok_matrix
from cyt import mwg_cauchy,mwg_tv, argumentspack
import copy
import math
import time
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import iradon_sart
import scipy.io
from timeit import timeit
from matrices import radonmatrix
from collections import namedtuple
from cplus import f
t = np.linspace(0,2*np.pi,50)
radonmatrix(164,t,7)
#f(1)

# K = argumentspack(M=1,y=2.0)
# L = copy.copy(K)
# L.M = 2
# print(K.M)

# STest = namedtuple("kokeilu", "a b")
# class koe:
#     def __init__(self):
#         self.a = 1.0
#         self.b = -2.0
#
# class koe2:
#     __slots__ = ['a', 'b']
#     def __init__(self):
#         self.a = 1.0
#         self.b = -2.0
#
# if __name__ == '__main__':
#     k = koe()
#     kk = koe2()
#     aa = STest(a=1.0, b=-2.0)
#     m = [1.0, -2.0]
#     l = (1.0,-2.0)
#     kk.b = 1110
#     k.c = 33
#     print(getsizeof(kk))
    #print  (timeit("z = k.a", setup="from __main__ import k"))
    #print(timeit("z = kk.a", setup="from __main__ import kk"))
    #print(timeit("z = aa[0]", setup="from __main__ import aa"))
    #print(timeit("z = aa.a", setup="from __main__ import aa"))
    # print(timeit("z = m[0]", setup="from __main__ import m"))


# K = lil_matrix((5,5))
# K[:,1] = np.array([[1,1,1,1,1]]).T
# exit(1)
# np.random.seed(1)
# M = sp.eye(400,format='csc')
# Lx = sp.eye(400,format='csc')
# Ly = sp.eye(400,format='csc')
# y = np.random.rand(400,1)
# x0 = np.random.rand(400,1)
# N = 20000
# t = time.time()
# d=mwg_tv(M, Lx, Ly,  y, x0,N, regalpha=2, samplebeta=0.3, sampsigma=1,lhsigma=1)
# print(time.time()-t)
# F = sp.load_npz('koe.npz')
# M = sp.load_npz('radonmatrix/full-57x50.npz')
# image = imread("shepp.png", as_gray=True)
# image = rescale(image, scale=0.1, mode='edge', multichannel=False)
# r = scipy.io.loadmat('r.mat')
# r = r['r']
# image = np.reshape(r,(57,57))
# theta = np.linspace(0., 179., 50, endpoint=True)
# ifl = np.reshape(image,(-1,1))
# ifl2 = np.reshape(image.T,(-1,1))
# R = radon(image,theta=theta,circle=False)
# M = sp.eye(40000,50000)
# N = sp.eye(50000)
# x = np.random.randn(50000,1)
# k = N.dot(x)
# j = M.dot(k)
# jj = M.dot(N).dot(x)
# # g = r-ifl
# F=F@ifl
# F = np.reshape(F,(81,50))
# M=M@ifl
# M = np.reshape(M,(81,50))
# plt.imshow(R)
# plt.figure()
# plt.imshow(F)
# plt.figure()
# plt.imshow(M)
# plt.show()
#
# exit(0)
# dim = 12
# d1= circulant(np.block([[2], [-1] , [np.zeros((dim - 3, 1))], [-1]]))
# regx2 = np.kron(np.eye(dim), d1)
# regy2 =  np.kron(d1, np.eye(dim))
#
# regvalues = np.array([2,-1,-1,-1,-1])
# offsets = np.array([0,1,-1,dim-1,-dim+1])
# reg1d = sp.diags(regvalues, offsets, shape=(dim, dim))
# regx = sp.kron(sp.eye(dim), reg1d)
# regy = sp.kron(reg1d,sp.eye(dim))
#
# reg1d = reg1d.toarray()
# regy=regy.toarray()
# regx = regx.toarray()
#exit(0)


# regvalues = np.array([1,-1,1])
# offsets = np.array([-dim+1,0,1])
# reg1d = sp.diags(regvalues, offsets, shape=(dim, dim))
# regx = sp.kron(sp.eye(dim), reg1d)
# regy = sp.kron(reg1d,sp.eye(dim))
#
# regy=regy.toarray()
# regx = regx.toarray()
# exit(0)

# T = 179
# R = 145
# M = np.zeros((R,T))
# print(gs(1.0,np.pi/4))
# t = np.linspace(0,np.pi,T,endpoint=True)
# r = np.linspace(-2.0,2.0,R,endpoint=True)
# tt = time.time()
# for i in range(0,R):
#     for j in range(0,T):
#         M[i, j] = gs(r[i], t[j])
#
# print(time.time()-tt)
# plt.imshow(M)
# plt.show()
# N = 77
# T = 100
# R = math.ceil(math.sqrt(2)*N)
# th = np.linspace(0, 179, T,endpoint=True)
# th_rad = th/360*2*np.pi
# # d = np.array([2,3,4])
# # c = np.array([0,1,0.0])
# # r = np.array([1,0,0])
# # M = coo_matrix((d, (r, c)), shape=(2,2))
#
# B = radonmatrix(N,th_rad)
# B = B.toarray()
# #M = coo_matrix((d, (r, c)), shape=(R*T,N*N))
# #M = csc_matrix(M)
# #exit(0)
# print('F')
# #fname = 'radonmatrix/'+ 'full-' +str(81) + 'x' + str(10) + '.npz'
# #M = sp.load_npz(fname)
# M = scipy.io.loadmat('k.mat')
# r = scipy.io.loadmat('r.mat')
# M = M['A']
# r = r['r']
#
# kk = B@r
# # jj = M@r
# #M = M.toarray()
# kk = np.reshape(kk,(R,T))
# # jj = np.reshape(jj,(R,T))
# r = np.reshape(r,(N,N))
# #scipy.io.savemat('arr.mat', mdict={'arr': M})
# #image = imread("shepp.png", as_gray=True)
# #image = rescale(image, scale=0.1, mode='edge', multichannel=False)
# R = radon(r,th,circle=False)
#
# #o = radon(image,l,circle=False)
# o = iradon_sart(R,th)
# koe = iradon_sart(kk,th)
# plt.imshow(o)
# plt.figure()
# plt.imshow(koe)
# plt.show()
# exit(0)

# sca = [0.05,0.1,0.2 ]
# nth = [10,25,50]
# for scaling in sca:
#     for N_theta in nth:
#         filename = "shepp.png"
#         image = imread(filename, as_gray=True)
#         image = rescale(image, scale=scaling, mode='edge', multichannel=False)
#         (dim, dimx) = image.shape
#         if (dim != dimx):
#             raise Exception('Image is not rectangular.')
#         N_r = math.ceil(math.sqrt(2) * dim)
#         theta = np.linspace(0., 179., N_theta, endpoint=True)
#         theta = theta/360*2*np.pi
#         flattened = np.reshape(image, (-1, 1))
#         #(N_r, N_theta) = (radon(image, theta, circle=False)).shape
#         fname = 'radonmatrix/'+ 'full-' +str(dim) + 'x' + str(N_theta) + '.npz'
#
#         if (not os.path.isfile(fname)):
#             #Mf = np.zeros([N_r * N_theta, dim * dim])
#             M = radonmatrix(dim,theta)
#             # M = lil_matrix((N_r*N_theta,dim*dim))
#             # empty = np.zeros([dim, dim])
#             # for i in range(0, dim):
#             #     for j in range(0, dim):
#             #         empty[i, j] = 1
#             #         #ww=np.ravel(np.reshape(radon(empty, theta, circle=False), (N_r * N_theta, 1)))
#             #         M[:, i * dim + j] = np.reshape(radon(empty, theta, circle=False), (N_r * N_theta, 1))
#             #         empty[i, j] = 0
#             # # qq = np.reshape(M@flattened,(N_r,N_theta))
#             # # plt.imshow(qq)
#             # # plt.figure()
#             # # plt.imshow(radon(image, theta, circle=True))
#             # # plt.show()
#             # # exit(1)
#             print(fname)
#             # M = csc_matrix(M)
#             sp.save_npz(fname,M)
            #np.savez_compressed(fname, radonoperator=M)

#cyt.mwg(10,np.zeros((10,1)) )
# def csr_spmul(Nrow,Ncol,data, indices, ptr, x):
#     Nrow = ptr.shape[0]-1
#     r  = np.zeros((Nrow,1))
#     for i in range(0,Nrow):
#         for j in range(ptr[i],ptr[i+1]):
#             r[i] += data[j] * x[indices[j]]
#
#     return r
#
# def csc_spmul(Nrow,Ncol,data, indices, ptr, x):
#     r  = np.zeros((Nrow,1))
#     for i in range(0,Ncol):
#         for j in range(ptr[i],ptr[i+1]):
#             r[indices[j]] += data[j] * x[i]
#
#     return r

# L = sp.eye(700000,700000,format="csc")
# y = 5*np.random.randn(700000,1)
# LL = csc_matrix(L)
# indices = LL.indices
# ptr = LL.indptr
# data = LL.data
# t = time.time()
# #q = csc_spmul(LL.shape[0],LL.shape[1],data,indices,ptr,y)
# q = csc_col(LL.shape[0],LL.shape[1],data,indices,ptr,3)
# print(time.time() - t)
# #print(q)
# t = time.time()
# qq = LL[:,3]
# print(time.time() - t)
# w = np.abs(qq-q)
# print(np.sum(w))

# @jit(nopython=True)
# def matrix(image, N_theta, N_r, dim):
#
#     theta = np.linspace(0., 180., N_theta, endpoint=False)
#     flattened = np.reshape(image, (-1, 1))
#     g = "G"
#     fname = "radonmatrix/" + str(N_r) + "x" + str(N_theta) + ".npz"
#
#     if (not os.path.isfile(fname)):
#         M = np.zeros([N_r * N_theta, dim * dim])
#         empty = np.zeros([dim, dim])
#         for i in range(0, dim):
#             for j in range(0, dim):
#                 empty[i, j] = 1
#                 M[:, i * dim + j] = np.ravel(
#                     np.reshape(radon(empty, theta, circle=True), (N_r * N_theta, 1)))
#                 empty[i, j] = 0
#         np.savez_compressed(fname, radonoperator=M)
#
# scale = 0.2
# image = imread("shepp.png", as_gray=True)
# image = rescale(image, scale=scale, mode='edge', multichannel=False)
# N_theta = 50
# theta = np.linspace(0., 180., N_theta, endpoint=False)
# (N_r, N_theta) = (radon(image, theta, circle=True)).shape
# (dim, dimx) = image.shape
# matrix(image,N_theta,N_r,dim)
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