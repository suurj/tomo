import os
import numpy as np
import scipy.sparse as sp
#import autograd_sparse as sp
#import autograd.numpy as np
from sys import getsizeof
from skimage.transform import radon, rescale,resize
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix, coo_matrix, dok_matrix
#from cyt import mwg_cauchy,mwg_tv, argumentspack
import copy
import math
import time
import scipy.signal as sg
import scipy.interpolate as interpolate
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import iradon_sart
import scipy.io
from timeit import timeit
from io import BytesIO
from matrices import radonmatrix
from collections import namedtuple
#from cplus import f
import cairosvg
import pathlib

from collections import defaultdict
class NestedDefaultDict(defaultdict):
        def __init__(self, *args, **kwargs):
            super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

        def __repr__(self):
            return repr(dict(self))


rr=NestedDefaultDict()
rr['3']['koe'][22] = 4.6
import json
json = json.dumps(rr)
f = open("dict.json","w")
f.write(json)
f.close()
exit(0)


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rcParams['text.latex.unicode'] = True
plt.rcParams['image.cmap'] ='gray'
rc('font', family='serif')

img = imread("drawing.png", as_gray=True)
##rr=radon(img, np.linspace(0,180,280), circle=False)

fig, ax = plt.subplots()
plt.imshow(img, interpolation='bilinear', aspect='auto',extent=[-1, 1, -1,1])
#[0, 180, np.sqrt(2), -np.sqrt(2)]
#ax.set_xlabel(r'time (s)',fontsize=12)
#ax.set_ylabel(r'Velocity', fontsize=12)
#ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
#             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=12, color='r')
plt.show()


# import h5py
#
# dir  = 'results/'
# fn = os.listdir(dir)
# fn = sorted(fn)
# fn = fn[-1]
# f = h5py.File(dir + fn, 'r')
# keys = list(f.keys())
# for i in range(keys.__len__()):
#     h = np.array(f[keys[i]])
#     print(keys[i],h)
# f.close()

exit(0)

class container:
    def __init__(self,target=np.zeros((20,20)),l1=-1.0,l2=-1.0,result=np.zeros((2,2)),noise=-1.0,imagefilename='None',targetsize=0,theta=0,method='None',prior='None',sampnum=0,adaptnum=0,alpha=0,globalprefix=""):
        self.spent = time.time()
        self.l1 = l1
        self.l2 = l2
        self.target = target
        self.noise = noise
        self.result = result
        self.imagefilename = imagefilename
        self.targetsize = targetsize
        self.theta = theta
        self.method = method
        self.prior = prior
        self.sampnum = sampnum
        self.adaptnum = adaptnum
        self.alpha = alpha
        self.globalprefix = globalprefix
        self.prefix = ''

    def finish(self,result=None,l1=None,l2=None):
        self.l1 = l1
        self.l2 = l2
        self.result = result
        self.spent = time.time()-self.spent
        self.prefix =  time.strftime("%Y-%b-%d_%H_%M_%S")

def correlationrow(M):
    if(len(M.shape)<=1 or M.shape[0]<=1):
        M = M - np.mean(M)
        M = scipy.signal.correlate(M, M, mode='full', method='fft')
        M = M[int((M.shape[0] - 1) / 2):]
        return  M/ M[0]

    else:
        M = M - np.mean(M,axis=1,keepdims=True)
        M = np.apply_along_axis(lambda x: scipy.signal.correlate(x,x, mode='full', method='fft'),axis=1,arr=M)
        M = M[:,int((M.shape[1] - 1) / 2):]
        return  M / np.reshape(M[:,0],(-1,1))


np.random.seed(10)
N = 3000
M = 30000
r = np.ones((M,N))
for i in range(1,N):
    r[:,i] = r[:,i-1] + 10.15*np.random.randn(M)

c = correlationrow(r[0:17,:])
cc=correlationrow(r[16,:])
# k = r[0,:]-np.mean(r[0,:])
# kk = r[0,0:-1:2]-np.mean(r[0,0:-1:2])
# tt = time.time()
# c = scipy.signal.correlate(k,k,mode='full',method='fft')
# cc = scipy.signal.correlate(kk,kk,mode='full',method='fft')
# c = c[int((c.shape[0]-1)/2):]
# c = c/c[0]
# cc = cc[int((cc.shape[0]-1)/2):]
# cc = cc/cc[0]
plt.plot(c[16,:])
plt.figure()
plt.plot(cc)
plt.show()

# r = container(adaptnum=1)
# r = None
# r.finish(result=np.zeros((50,50)))
# for attr, value in r.__dict__.items():
#         print (attr, value)
#
#
# def saveresult(result):
#     import h5py
#     with h5py.File( "koe2" + ".hdf5", 'w') as f:
#         for key, value in r.__dict__.items():
#             if (value is None):
#                 value = "None"
#             if (isinstance(value, np.ndarray)):
#                 compression = 'lzf'
#                 print('gg')
#             else:
#                 compression = None
#             f.create_dataset(key, data=value,compression=compression)
#     f.close()

# f = h5py.File('koe2.hdf5', 'r')
# keys = list(f.keys())
# print( keys)
# for i in range(keys.__len__()):
#     h = np.array(f[keys[i]])
#     print(h)
# f.close()


# from tqdm import tqdm
# r = np.random.randn(50,50)
# bar = tqdm(total=1000,leave=True)
# t = time.time()
# for i in range(0,1000):
#     #pass
#     bar.update(1)
#     time.sleep(0.01)
# bar.close()
# time.sleep(0.01)
# print(time.time()-t)
# exit(0)
from matrices import radonmatrix
# def gs(p,t):
#     M_PI =np.pi
#     M_SQRT2 = np.sqrt(2)
#     if (p<0):
#         p = -p
#
#     t = (t % (M_PI/2.0))
#     if(t >= M_PI/4.0):
#         t = M_PI/2.0-t
#
#     if( p > M_SQRT2):
#         a = 0
#         return a
#     else:
#         x1m = p/np.cos(t) + np.tan(t)
#         x1 = p/np.cos(t) - np.tan(t)
#         y1 = p/np.sin(t) - 1.0/np.tan(t)
#
#
#         if (x1 < 1.0 and x1m  < 1.0):
#             a = np.sqrt(4.0+(x1-x1m)*(x1-x1m))
#             return a
#
#         elif (x1 < 1.0 and x1m  > 1.0):
#             a = np.sqrt((1.0-x1)*(1.0-x1) + (1.0-y1)*(1.0-y1))
#             return a
#
#         elif (x1 >=1.0):
#             a = 0.0
#             return a
#
#         else:
#             return -9.0

# tt = np.zeros((50,50))
# theta = np.linspace(np.pi,-np.pi,endpoint=False)
# rh = np.linspace(-np.sqrt(2),np.sqrt(2),50,endpoint=True)
# for t in range(50):
#     for r in range (50):
#         tt[r,t] = gs(rh[r],theta[t])
# N = 64#541
# N_big = 128
# N_theta = 50#127
# N_thetabig = 50
# fname = 'radonmatrix/' + 'full-' + str(N_big) + 'x' + str(N_thetabig) + '.npz'
# theta=np.linspace(0.,180., N_theta, endpoint=False)
# theta=theta/360*2*np.pi
# N_r_big = math.ceil(np.sqrt(2) * N_big)
# N_r = math.ceil(np.sqrt(2) * N)
# rhoobig = np.linspace(-np.sqrt(2), np.sqrt(2), N_r_big, endpoint=True)
# rhoo = np.linspace(-np.sqrt(2), np.sqrt(2), N_r, endpoint=True)
# #
# image=imread(BytesIO(cairosvg.svg2png(url="big.svg",output_width=128,output_height=128)),as_gray=True)
# image2 = resize(image, (64, 64),anti_aliasing=False,preserve_range=True,order=1,mode='symmetric')
# flattened = np.reshape(image,(-1,1))
# noise = 0.01
#
# radonoperatorbig = sp.load_npz(fname)/N_big
# simulated = radonoperatorbig @ flattened
# maxsim = np.max(simulated)
# simulated = simulated + maxsim*noise*np.random.randn(N_r_big * N_thetabig, 1)
# sgramsim = np.reshape(simulated,(N_r_big,N_thetabig))
# #xx, yy = np.meshgrid(theta, rhoobig)
# interp = interpolate.RectBivariateSpline(rhoobig,theta,sgramsim)
# # xnew, ynew = np.meshgrid( rhoo,theta)
# # xnew = np.reshape(xnew,(-1,))
# # ynew = np.reshape(ynew,(-1,))
# lines = interp( rhoo,theta)
# sgram = lines
#sgram = np.reshape(lines,(N_r,N_theta))


#radonoperator=radonmatrix(N, theta)
# radonoperator = sp.load_npz(fname)/N
# flattened = np.reshape(image, (-1, 1))
# measurement = radonoperator @ flattened
# sgram = np.reshape(measurement,(math.ceil(np.sqrt(2)*N),N_theta))
# #sgram = radon(image,theta/(2*np.pi)*360,circle=False)
# #radonoperator= radonmatrix(N, theta)
#sp.save_npz(fname,radonoperator)
# sgram2 = radon(image,theta/(2*np.pi)*360,circle=False)
# plt.imshow(sgramsim,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# #plt.imshow(sgram,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# plt.figure()
# plt.imshow(sgram2,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# # plt.imshow(sgram2,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# # #plt.imshow(tt,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# plt.figure()
# plt.imshow(sgram,extent=[theta[0], theta[-1], -np.sqrt(2), np.sqrt(2)])
# # plt.imshow(iradon_sart(sgram2,theta/(2*np.pi)*360))
# plt.show()
# exit(0)
# def matern(N,l1,l2,alpha):
#     N = int(N)
#     L1 = np.ones((N,N))
#     L1 = np.ravel(L1)
#     L2 = np.ones((N, N))
#     L2 = np.ravel(L2)
#     r = np.arange(0,N)
#     r = np.linspace(-1,1,N,endpoint=True)
#     xi = np.tile(r,(N,1))
#     xif = np.reshape(xi,(-1,1))
#     xif = np.ravel(xif)
#     yi = np.flip(np.copy(xi.T),axis=0)
#     yif = np.reshape(yi, (-1, 1))
#     yif = np.ravel(yif)
#     #M[np.abs(xif)<0.5] = np.where(xif**2)
#     #M=np.where((np.abs(xif)<0.5) * (yif < 0) >=0,xif**2,M)
#     #M = np.where((xif**2 + yif**2/3 <0.25) , 5, M)
#     #L1 = np.where((l1(xif,yif)), l1as(xif,yif), L1)
#     L1=np.vectorize(l1)(xif, yif)
#     #L2 = np.where((l2(xif, yif)), l2as(xif, yif), L2)
#     L2 = np.vectorize(l2)(xif,yif)
#     L1f = L1.reshape((-1,1))
#     L2f = L2.reshape((-1,1))
#     a = np.vectorize(alpha)(xif,yif)
#     #a = 1#np.where((l1(xif,yif)), l1as(xif,yif), L1)
#     #M[N/2,N/2] = 0
#     #M = np.reshape(M,(N,N))
#
#     regvalues = np.array([2, -1, -1, -1, -1])
#     offsets = np.array([0, 1, -1, N - 1, -N + 1])
#     reg1d = sp.diags(regvalues, offsets, shape=(N, N))
#     #reg1d = reg1d.toarray()
#     regx = sp.kron(sp.eye(N), reg1d)
#     regy = sp.kron(reg1d, sp.eye(N))
#     #regy = regy.toarray()
#     #regx = regx.toarray()
#     H =  sp.eye(N*N)+regy.multiply( L1[:, np.newaxis]) +  regx.multiply( L2[:, np.newaxis])
#     #H = H.toarray()
#     Gd = np.ravel(a)*np.sqrt(np.ravel(L1f*L2f))
#     Ginv = sp.diags(1/Gd)
#     S =  ((H.T).dot(Ginv)).dot(H)
#     S = S.toarray()
#     return S
#
# def l(x,y):
#     if (x *x  + y *y  / 3 < 0.25):
#         return 2
#     else:
#         return 1
#
# def alpha(x,y):
#     return 10
#
# #alpha =lambda x,y: 1
# #l1 = lambda x,y: x**2 + y**2/3 <0.25
# #l1as = lambda x,y: 2
# t = time.time()
# f=matern(32,l,l,alpha)
# print(time.time()-t)
# plt.imshow(f)
# plt.show()
# exit(0)
#
# def  gradient(f,Q,x):
#     N = x.shape[0]
#     eps = 1e-7;
#     xn = x.copy(); xe = x.copy();
#     gr = np.zeros((N,1));
#     for i  in range(N):
#         xn[i] = x[i] + eps;
#         xe[i] = x[i] - eps;
#         gr[i] = (f(xn,Q) -f(xe,Q))/(2*eps);
#         xn[i] = x[i];
#         xe[i] = x[i];
#
#     return  gr
#
# from cyt import tfun_cauchy, mwg_cauchy , mwg_tv,tfun_tikhonov, tikhonov_grad,tfun_tv, tv_grad ,cauchy_grad,argumentspack
# Q = argumentspack()
# Q.s2 = 1
# Q.b = 0.3
# Q.M = np.array([[3, -1], [-1, 3]])
# Q.M = np.linalg.inv(Q.M)
# Q.M = np.linalg.cholesky(Q.M)
# Q.M = Q.M.T
# Q.Lx = np.zeros((1,2))
# Q.Ly=np.zeros((1,2))
# Q.Ly = csc_matrix(Q.Ly)
# Q.Lx = csc_matrix(Q.Lx)
# Q.y = np.array([[2.0,-50.0]]).T
# Q.y = Q.M.dot(Q.y)
# np.random.seed(1)
# x = np.random.randn(2,1)
# t = time.time()
# nadapt = 500
# gg = mwg_tv(25000,nadapt,Q,x , sampsigma=1,cmesti=False)
# print(time.time()-t)
# print(np.cov(gg[:,nadapt:]))
# print(np.mean(gg[:,nadapt:],axis=1))
# plt.plot(gg[0,:],gg[1,:],'*r')
# plt.show()
# r = cauchy_grad(x,Q)
# rr = gradient(tfun_cauchy,Q,x)
# w = tikhonov_grad(x,Q)
# ww = gradient(tfun_tikhonov,Q,x)
# g = tv_grad(x,Q)
# gg = gradient(tfun_tv,Q,x)
# print(r-rr,gg-g,ww-w)
# print(timeit("z = np.dot(Q.M.T,Q.M)", number=10000 ,setup="from __main__ import  Q,x, np"))
# print(timeit("z = Q.M.T.dot(Q.M)", number=10000 ,setup="from __main__ import  Q,x"))

# t = np.linspace(0,2*np.pi,50)

# def gradi(x,A):
#     return -0.5*2*((A[0].T).dot(A[0])).dot(x)
#     #*np.exp(-0.5*np.dot(res.T,res))
#     #return np.reshape(-np.matmul(np.matmul(A[0].T,A[0]),x),(-1,1))
#
#
# def logdensity(theta,A):
#     #print(A[0].dot(theta))
#     #print(-1* np.dot(A[0].dot(theta).T, A[0].dot(theta)),theta)
#     return -0.5* np.dot(A[0].dot(theta).T, A[0].dot(theta))
#     #return 2 * np.dot((np.dot(A[0], theta)).T , (np.dot(A[0], theta)))
#     #return np.exp(-0.5*(np.dot(A,theta)).T@(np.dot(A,theta)) - 0.5 * np.dot(r.T, r))
#     #return np.exp(np.log(pdf(theta)) - 0.5 * np.dot(r.T, r))
#     #return np.exp(np.log(pdf(theta))-0.5*np.dot(r.T,r))

# radonmatrix(16,t,1)
# #f(1)

# K = argumentspack(M=1,y=2.0)
# L = copy.copy(K)
# L.M = 2
# print(K.M)

# STest = namedtuple("kokeilu", "a b")
# def rr(x):
#     return x**2.0
# class koe:
#     def __init__(self):
#         self.a = 1.0
#         self.b = -2.0
#
#     def pp(self,x):
#         return x**2.0
#
# class koe2:
#     __slots__ = ['a', 'b']
#     def __init__(self):
#         self.a = 1.0
#         self.b = -2.0
#
#     def pp(self, x):
#         return x ** 2.0
#
# #if __name__ == '__main__':
# k = koe()
# kk = koe2().pp
# m = [1.0, -2.0]
# l = (1.0,-2.0)
# #kk.b = 1110
# #k.c = 33
# #print(getsizeof(kk))
# print  (timeit("z = k.pp(3)", number=1000000, setup="from __main__ import k"))
#

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