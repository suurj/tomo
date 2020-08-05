from skimage.io import imread
from skimage.transform import radon, resize
import warnings
import numpy as np
import pywt
import scipy.interpolate as interpolate
from scipy.optimize import minimize
from scipy.signal import correlate
import time
import math
import sys
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
import argparse
import pathlib
from tqdm import tqdm
from cyt import tfun_cauchy as lcauchy, tfun_tikhonov as ltikhonov, tikhonov_grad, tfun_tv as ltv, tv_grad, cauchy_grad, isocauchy_grad, argumentspack

# Class to store results of one computation.
class container:
    def __init__(self,target=np.zeros((2,2)),l1=-1.0,l2=-1.0,result=np.zeros((2,2)),thinning=-1,noise=-1.0,imagefilename=None,targetsize=0,theta=np.zeros((1,)),method=None,prior=None,crimefree=False,totaliternum=0,levels=0,adaptnum=0,alpha=0.0,globalprefix=""):
        self.spent = time.time()
        self.l1 = l1
        self.l2 = l2
        self.target = target
        self.noise = noise
        self.crimefree = crimefree
        self.result = result
        self.imagefilename = imagefilename
        self.targetsize = targetsize
        self.theta = theta
        self.thinning = thinning
        self.method = method
        self.prior = prior
        self.totaliternum = totaliternum
        self.adaptnum = adaptnum
        self.alpha = alpha
        self.levels = levels
        self.globalprefix = globalprefix
        self.chain = None
        self.prefix = ''

    def intermedfilename(self):
        return self.globalprefix + '/' + time.strftime("%Y-%b-%d_%H_%M_%S") + '+' + self.prior + '+' + self.method + '+' + str(self.noise) + '+' + str(self.theta[0])+ '_' + str(self.theta[-1]) + '-'  + str(self.targetsize) + 'x' + str(len(self.theta))


    def finish(self,result=None,chain=None,error=(-1.0,-1.0),iters=None,thinning=-1):
        self.l1 = error[0]
        self.l2 = error[1]
        if iters is not None:
            self.totaliternum = iters
        if (chain is not None):
            self.thinning = thinning
            self.chain=chain
        self.result = result
        self.spent = time.time()-self.spent
        self.prefix =  time.strftime("%Y-%b-%d_%H_%M_%S") + '+' + self.prior + '+' + self.method + '+' + str(self.noise) + '+' + str(self.theta[0])+ '_' + str(self.theta[-1]) + '-'  + str(self.targetsize) + 'x' + str(len(self.theta))

class tomography:

    def __init__(self, filename="shepp.png", targetsize=128, itheta=50, noise=0.0,  commonprefix="", dimbig = 599, N_thetabig=421, crimefree=False,lhdev=None,dataload=False):
        if dataload is False:
            self.globalprefix = str(pathlib.Path.cwd()) + commonprefix
            if not os.path.exists(self.globalprefix):
                os.makedirs(self.globalprefix)
            self.filename = filename
            self.dim = targetsize
            self.noise = noise
            self.thetabig = None
            self.dimbig = dimbig
            self.N_thetabig = N_thetabig
            self.N_rbig = math.ceil(np.sqrt(2) * self.dimbig)
            self.crimefree = crimefree
            if targetsize > 1100:
                raise Exception(
                    'Dimensions of the target image are too large (' + str(targetsize) + 'x' + str(targetsize) + ')')

            img = self.opendata(filename)
            (dy, dx) = img.shape
            if (dy != dx):
                raise Exception('Image is not rectangular.')

            self.targetimage = self.opendata(filename)
            self.targetimage = resize(self.targetimage, (self.dim, self.dim), anti_aliasing=False, preserve_range=True,
                                order=1, mode='symmetric')
            if self.crimefree:
                image = self.opendata(filename)
                image = resize(image, (self.dimbig, self.dimbig), anti_aliasing=False, preserve_range=True,
                                    order=1, mode='symmetric')
                (self.simsize, _) = image.shape

            else:
                image = self.opendata(filename)
                image = resize(image, (targetsize, targetsize), anti_aliasing=False, preserve_range=True, order=1,
                                    mode='symmetric')
                (self.simsize, _) = image.shape

            self.flattened = np.reshape(image, (-1, 1))

            if isinstance(itheta, (int, np.int32, np.int64)) or (isinstance(itheta,(list,tuple)) and len(itheta) == 1):
                if  isinstance(itheta,(list,tuple)):
                    itheta = itheta[0]
                self.theta = np.linspace(0., 180., itheta, endpoint=False)
                self.theta = self.theta / 360 * 2 * np.pi
                (self.N_r, self.N_theta) = (math.ceil(np.sqrt(2) * self.dim), itheta)
                self.rhoo = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_r, endpoint=True)
                fname = 'radonmatrix/0_180-{0}x{1}.npz'.format(str(self.dim), str(self.N_theta))

                if self.crimefree:
                    self.thetabig = np.linspace(0, 180, self.N_thetabig, endpoint=False)
                    self.thetabig = self.thetabig / 360 * 2 * np.pi
                    self.rhoobig = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_rbig, endpoint=True)

            elif len(itheta) == 3:
                self.theta = np.linspace(itheta[0], itheta[1], itheta[2], endpoint=False)
                self.theta = self.theta / 360 * 2 * np.pi
                (self.N_r, self.N_theta) = (
                    math.ceil(np.sqrt(2) * targetsize), itheta[2])
                self.rhoo = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_r, endpoint=True)
                fname = 'radonmatrix/{0}_{1}-{2}x{3}.npz'.format(str(itheta[0]), str(itheta[1]), str(self.dim), str(self.N_theta))

                if (self.crimefree):
                    self.thetabig = np.linspace(itheta[0], itheta[1], self.N_thetabig, endpoint=False)
                    self.thetabig = self.thetabig / 360 * 2 * np.pi
                    self.rhoobig = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_rbig, endpoint=True)

            else:
                raise Exception('Invalid angle input.')

            if not os.path.isfile(fname):
                path = os.path.dirname(os.path.abspath(fname))
                if not os.path.exists(path):
                    os.makedirs(path)

                from matrices import radonmatrix
                self.radonoperator = radonmatrix(self.dim, self.theta)
                sp.save_npz(fname, self.radonoperator)

            # In the case of inverse-crime free tomography,
            # one might use the Radon tool from scikit-image
            # or construct another Radon matrix and calculate a sinogram with that. The former is definitely faster and also preferred,
            # since different methods are used to simulate and reconcstruct the image.

            self.radonoperator = sp.load_npz(fname)
            self.radonoperator = sp.csc_matrix(self.radonoperator)
            self.radonoperator = self.radonoperator / self.dim

            if self.crimefree:
                #self.radonoperatorbig = sp.load_npz(fnamebig) / self.dimbig
                #simulated = self.radonoperatorbig@self.flattened
                #simulated = np.reshape(simulated,(self.N_rbig,self.N_thetabig))
                simulated = self.radonww(image,self.thetabig/ ( 2 * np.pi)*360)/self.dimbig
                simulated = np.reshape(simulated,(-1,1))

                maxvalue = np.max(simulated)
                simulated = simulated + maxvalue * self.noise * np.random.randn(self.N_rbig * self.N_thetabig, 1)
                self.sgramsim = np.reshape(simulated, (self.N_rbig, self.N_thetabig))
                interp = interpolate.RectBivariateSpline(-self.rhoobig, self.thetabig, self.sgramsim,kx=1,ky=1)
                self.sgram = interp(-self.rhoo, self.theta)
                self.lines = np.reshape(self.sgram, (-1, 1))

            else:
                simulated = self.radonoperator @ self.flattened
                maxvalue = np.max(simulated)
                noiserealization = np.random.randn(self.N_r * self.N_theta, 1)
                self.lines = simulated + maxvalue * self.noise * noiserealization
                self.sgram = np.reshape(self.lines, (self.N_r, self.N_theta))

            if lhdev is None:
                if self.noise == 0:
                    lhdev = 0.01
                else:
                    lhdev = self.noise
            self.lhsigmsq = (maxvalue * lhdev) ** 2
            self.Q = argumentspack(M=self.radonoperator, y=self.lines, b=0.01, s2=self.lhsigmsq)
            self.pbar = None
            self.method =   'L-BFGS-B'

        else:
            self.crimefree = True
            self.noise = 0.05
            self.filename = ""
            self.targetimage = ""
            self.dim = None
            self.globalprefix =""
            self.theta = 360

    def opendata(self,fname):
            if fname.endswith(('.mat')):
                import scipy.io
                image = np.array(scipy.io.loadmat(fname)['A'])
            else:
                image=imread(fname, as_gray=True)
            return image

    def mincb(self,_):
        self.pbar.update(1)

    def dataload(self,Mfile,Mname,dfile,dname,scaling=1,imsize=128):
        import scipy.io
        try:
            matrix = scipy.io.loadmat(Mfile)[Mname]
            sino = scipy.io.loadmat(dfile)[dname]
        except:
            import h5py

            with h5py.File(dfile, 'r') as f:
                a = list(f.keys())
                sino = np.double(np.array(f[dname]))
                s = sino.shape[0]*sino.shape[1]

            with h5py.File(Mfile, 'r') as f:
                a = list(f.keys())
                data = f[Mname]['data']
                ir = f[Mname]['ir']
                jc = f[Mname]['jc']
                matrix = sp.csc_matrix((data, ir, jc),shape=(s,imsize**2))

        plt.imshow(sino)
        plt.show()
        self.radonoperator =  sp.csc_matrix(matrix)
        self.dim = np.int(np.sqrt(self.radonoperator.shape[1]))
        self.sgram = np.double(sino)
        self.lines = np.reshape(self.sgram,(-1,1),order='F')*scaling
        self.lhsigmsq = (np.max(self.lines) * 0.05) ** 2
        self.Q = argumentspack(M=self.radonoperator, y=self.lines, b=0.001, s2=self.lhsigmsq)
        self.pbar = None
        self.method = 'L-BFGS-B'


    def map_tikhonov(self, alpha=1.0, order=1,maxiter=400,retim=True):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree, prior='tikhonov', levels=order, method='map', noise=self.noise, imagefilename=self.filename,
                            target=self.targetimage, targetsize=self.dim,globalprefix=self.globalprefix, theta=self.theta/(2*np.pi)*360)
        if (order == 2):
            regN = np.diag([2] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1) + np.diag([-1] * (self.dim - 1), -1);
            regN = regN[1:-1, :]
            regN = sp.csc_matrix(regN);
            # regvalues = np.array([2, -1, -1, -1, -1])
            # offsets = np.array([0, 1, -1, self.dim - 1, -self.dim + 1])
            # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        else:
            regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
            regN = sp.csc_matrix(regN[0:-1, :])
            # regvalues = np.array([1, -1, 1])
            # offsets = np.array([-self.dim + 1, 0, 1])
            # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
            # reg1d = sp.csc_matrix(reg1d)
            # reg1d[self.dim-1,self.dim-1] = 0
            # print(np.linalg.matrix_rank(reg1d.todense()))
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        print("Running MAP estimate for Tikhonov prior.")
        self.pbar = tqdm(total=np.Inf,file=sys.stdout)
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        solution = minimize(self.tfun_tikhonov, x0, method=self.method, jac=self.grad_tikhonov,
                            options={'maxiter': maxiter, 'disp': False},callback=self.mincb)
        self.pbar.close()
        iters = solution.nit
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution,error=self.difference(solution),iters=iters)
            return res
        else:
            return solution

    def tfun_tikhonov(self, x):
        return -ltikhonov(x, self.Q)

    def grad_tikhonov(self, x):
        x = x.reshape((-1, 1))
        ans = -tikhonov_grad(x, self.Q)
        return np.ravel(ans)

    def map_tv(self, alpha=1.0, maxiter=400,retim=True):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree, prior='tv', method='map', noise=self.noise, imagefilename=self.filename,
                            target=self.targetimage, targetsize=self.dim,globalprefix=self.globalprefix, theta=self.theta/(2*np.pi)*360)
        regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
        regN = sp.csc_matrix(regN[0:-1, :])
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        #total = np.vstack((self.radonoperator.todense(),combined.todense()))
        #print(total)
        #print(total.shape)
        #print(np.linalg.matrix_rank(total))
        #exit(0)
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        print("Running MAP estimate for TV prior.")
        self.pbar = tqdm(total=np.Inf,file=sys.stdout)
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        solution = minimize(self.tfun_tv, x0, method=self.method, jac=self.grad_tv,
                            options={'maxiter': maxiter, 'disp': False},callback=self.mincb)
        self.pbar.close()
        iters = solution.nit
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution,error=self.difference(solution),iters=iters)
            return res
        else:
            return solution

    def tfun_tv(self, x):
        return -ltv(x, self.Q)

    def grad_tv(self, x):
        x = x.reshape((-1, 1))
        q = -tv_grad(x, self.Q)
        return np.ravel(q)

    def map_cauchy(self, alpha=0.05, maxiter=400,retim=True,isotropic=False):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree,prior='cauchy',method='map',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)

        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        if(isotropic==False):
            regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
            regN = sp.csc_matrix(regN[0:-1, :])
            help = np.zeros((2, self.dim));
            help[0, 0] = 1;
            help[1, self.dim - 1] = 1
            help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
            # regvalues = np.array([1, -1, 1])
            # offsets = np.array([-self.dim + 1, 0, 1])
            # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
            # reg1d = sp.csc_matrix(reg1d)
            # reg1d[self.dim-1,self.dim-1] = 0
            # print(np.linalg.matrix_rank(reg1d.todense()))
            # regx = sp.kron(sp.eye(self.dim), reg1d)
            # regy = sp.kron(reg1d, sp.eye(self.dim))
            # regx = sp.csc_matrix(regx)
            # regy = sp.csc_matrix(regy)
            regx = sp.kron(sp.eye(self.dim), regN)
            regy = sp.kron(regN, sp.eye(self.dim))
            regx = sp.csc_matrix(regx)
            regy = sp.csc_matrix(regy)
            regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
            regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
            # combined = sp.vstack([regy, regx], format='csc')
            combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
            empty = sp.csc_matrix((1, self.dim * self.dim))
            self.Q.Lx = combined
            self.Q.Ly = empty
            self.Q.a = alpha
            self.Q.s2 = self.lhsigmsq
            self.Q.b = 0.01

        else:
            dim = self.dim
            regvalues = np.array([1, -1, 1])
            offsets = np.array([-dim + 1, 0, 1])
            reg1d = sp.diags(regvalues, offsets, shape=(dim, dim))
            reg1d = sp.csc_matrix(reg1d)
            reg1d[dim - 1, dim - 1] = 0
            regx = sp.kron(sp.eye(dim), reg1d)
            regy = sp.kron(reg1d, sp.eye(dim))
            regx = sp.csc_matrix(regx)
            regy = sp.csc_matrix(regy)

            rmxix = np.sum(np.abs(regx), axis=1) == 1
            rmyix = np.sum(np.abs(regy), axis=1) == 1
            boundary = rmxix + rmyix
            regx[np.ravel(boundary), :] = 0
            regy[np.ravel(boundary), :] = 0

            bmatrix = sp.csc_matrix((dim * dim, dim * dim))
            q = np.where((np.ravel(boundary) == True))
            bmatrix[q, q] = 1

            self.Q.Lx = regx
            self.Q.Ly = regy
            self.Q.a = alpha
            self.Q.boundarya = alpha
            self.Q.s2 = self.lhsigmsq
            self.Q.b = 0.01
            self.Q.boun = bmatrix



        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        print("Running MAP estimate for Cauchy prior.")
        self.pbar = tqdm(total=np.Inf,file=sys.stdout)

        if(isotropic==False):
            solution = minimize(self.tfun_cauchy, x0, method=self.method, jac=self.grad_cauchy,
                            options={'maxiter': maxiter, 'disp': False},callback=self.mincb)
        else:
            solution = minimize(self.tfun_isocauchy, x0, method=self.method, jac=self.grad_isocauchy,
                                options={'maxiter': maxiter, 'disp': False}, callback=self.mincb)
        self.pbar.close()
        iters = solution.nit
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution,error=self.difference(solution),iters=iters)
            return res
        else:
            return solution

    def tfun_cauchy(self, x):
        return -lcauchy(x, self.Q)


    def grad_cauchy(self, x):
        x = x.reshape((-1, 1))
        ans = -cauchy_grad(x, self.Q)
        return (np.ravel(ans))

    def tfun_isocauchy(self,x):
        x = np.reshape(x, (-1, 1))
        x = np.reshape(x, (-1, 1))
        Mxy = self.Q.M.dot(x) - self.Q.y
        Lxx = self.Q.Lx.dot(x)
        Lyx = self.Q.Ly.dot(x)
        B = self.Q.boun.dot(x)
        alpha = self.Q.a
        alphab = self.Q.boundarya
        return -(-0.5 / self.Q.s2 * Mxy.T.dot(Mxy) - 3 / 2 * np.sum(np.log(alpha + np.multiply(Lxx, Lxx) + np.multiply(Lyx, Lyx))) - np.sum(np.log(alphab + np.multiply(B, B))))

    def grad_isocauchy(self,x):
        x = x.reshape((-1, 1))
        gr = np.ravel(isocauchy_grad(x,self.Q))
        q=np.abs(gr+self.grad_isocauchy2(x))
        print(np.max(q))
        return -gr

    def grad_isocauchy2(self,x):
        x = x.reshape((-1, 1))
        M = self.Q.M
        Lx = self.Q.Lx
        Lxx = self.Q.Lx.dot(x)
        Ly = self.Q.Ly
        Lyx = self.Q.Ly.dot(x)
        alpha = self.Q.a
        B = self.Q.boun
        Bx = B.dot(x)
        s2 = self.Q.s2
        y = self.Q.y
        alphab = self.Q.boundarya
        Mxy = M.dot(x) - y
        gr = np.ravel(-1.0 / s2 * (M.T).dot(Mxy))
        t1 = alpha + (np.multiply(Lxx, Lxx)) + (np.multiply(Lyx, Lyx))
        # t2 = (np.multiply(Lyx, Lyx))
        t2 = alphab + np.multiply(Bx, Bx)
        a = np.ones((1, Lxx.shape[0]))
        gr += np.ravel(-3 / 2 * a @ (((2 * sp.diags(np.ravel(Lxx), format='csc')) @ Lx + (
                    2 * sp.diags(np.ravel(Lyx), format='csc')) @ Ly) / (t1))) + np.ravel(
            - a @ ((2 * sp.diags(np.ravel(Bx), format='csc')) @ B / (t2)))
        # gr =  np.ravel(gr)  -3/2*np.sum(((2*sp.diags(np.ravel(Lxx),format='csc'))@Lx + (2*sp.diags(np.ravel(Lyx),format='csc'))@Ly)/(alpha+t1+t2),axis=0) - np.sum( (2*sp.diags(np.ravel(Bx),format='csc'))@B/(alphab+t3),axis=0)
        return -gr

    def map_wavelet(self, alpha=1.0, type='haar', maxiter=400,levels=None ,retim=True):
        res = None
        if (levels is None):
            levels = int(np.floor(np.log2(self.dim))-1)
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree,prior=type,method='map',levels=levels,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from matrices import totalmatrix
        wl = pywt.Wavelet(type)
        g = np.array(wl.dec_lo)
        h = np.array(wl.dec_hi)
        regx = totalmatrix(self.dim, levels, g, h)
        regy = sp.csc_matrix((1, self.dim * self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        self.Q.Lx = regx
        self.Q.Ly = regy
        self.Q.a = alpha
        self.Q.b = 0.01
        self.Q.s2 = self.lhsigmsq
        print("Running MAP estimate for Besov prior (" + type + ' '  + str(levels) + ').' )
        self.pbar = tqdm(total=np.Inf,file=sys.stdout)
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        solution = minimize(self.tfun_tv, x0, method=self.method, jac=self.grad_tv,
                            options={'maxiter': maxiter, 'disp': False},callback=self.mincb)
        self.pbar.close()
        iters = solution.nit
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution), iters=iters)
            return res
        else:
            return solution

    def hmcmc_tikhonov(self, alpha, M=100, Madapt=20, order=1,mapstart=False,thinning=1,retim=True,variant='hmc',interstep=100):
        res = None
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tikhonov',method=variant,levels=order,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc, ehmc
        if (order == 2):
            regN = np.diag([2] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1) + np.diag([-1] * (self.dim - 1), -1); regN = regN[1:-1,:]
            regN = sp.csc_matrix(regN);
            # regvalues = np.array([2, -1, -1, -1, -1])
            # offsets = np.array([0, 1, -1, self.dim - 1, -self.dim + 1])
            # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        else:
            regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
            regN = sp.csc_matrix(regN[0:-1, :])
            # regvalues = np.array([1, -1, 1])
            # offsets = np.array([-self.dim + 1, 0, 1])
            # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
            # reg1d = sp.csc_matrix(reg1d)
            # reg1d[self.dim-1,self.dim-1] = 0
            # print(np.linalg.matrix_rank(reg1d.todense()))
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        #p=np.array(combined.todense())
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.001
        self.Q.logdensity = ltikhonov
        self.Q.gradi = tikhonov_grad
        self.Q.y = self.lines
        if (mapstart):
            x0 = np.reshape(self.map_tikhonov(alpha,maxiter=150),(-1,1))
            x0 = x0 + 0.000001*np.random.rand(self.dim*self.dim,1)
        else:
            x0 = 0.0 + 0.00*np.random.randn(self.dim * self.dim, 1)
        print("Running  " + variant.upper() + " for Tikhonov prior.")
        if (variant == 'hmc'):
            solution, chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75,cmonly=retim, thinning=thinning,istep=interstep,intername=res.intermedfilename)
        else:
            solution, chain = ehmc(M, x0, self.Q, Madapt, kappa=0.75, cmonly=retim,thinning=thinning,stepsize=0.002,istep=interstep,intername=res.intermedfilename)
        #solution,chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_tv(self, alpha, M=100, Madapt=20,mapstart=False,thinning=1,retim=True,variant='hmc',interstep=100):
        res = None
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tv',method=variant,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc, ehmc
        regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
        regN = sp.csc_matrix(regN[0:-1, :])
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.0001
        self.Q.logdensity = ltv
        self.Q.gradi = tv_grad
        if mapstart:
            x0 = np.reshape(self.map_tv(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.00001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.0 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running " + variant.upper() + " for TV prior.")
        if (variant == 'hmc'):
            solution, chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75,cmonly=retim, thinning=thinning,istep=interstep,intername=res.intermedfilename)
        else:
            solution, chain = ehmc(M, x0, self.Q, L=50, Madapt=Madapt,  cmonly=retim,thinning=thinning,stepsize=0.002,istep=interstep,intername=res.intermedfilename)
        #solution,chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_cauchy(self, alpha, M=100, Madapt=20,thinning=1,mapstart=False,retim=True,variant='hmc',interstep=100):
        res = None
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='cauchy',method=variant,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc, ehmc
        regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
        regN = sp.csc_matrix(regN[0:-1, :])
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        self.Q.logdensity = lcauchy
        self.Q.gradi = cauchy_grad
        if mapstart:
            x0 = np.reshape(self.map_cauchy(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.00001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.0 + 0.00*np.random.randn(self.dim * self.dim, 1)
        print("Running " + variant.upper() + " for Cauchy prior.")
        #solution,chain = nonuts_hmc(M, x0, self.Q, 10, L=100, delta=0.65,cmonly=False, thinning=thinning)
        #solution,chain = ehmc(M, x0, self.Q, epstrials=25,Ltrials=25, L=50, delta=0.65,cmonly=False, thinning=thinning)
        #solution = np.median(chain,axis=1)
        if(variant=='hmc'):
            solution,chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cmonly=retim, thinning=thinning,istep=interstep,intername=res.intermedfilename)
        else:
            solution,chain = ehmc(M, x0, self.Q, Madapt, cmonly=retim, thinning=thinning,istep=interstep,intername=res.intermedfilename)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_wavelet(self, alpha, M=100, Madapt=20, type='haar',levels=None,mapstart=False,thinning=1,retim=True,variant='hmc',interstep=100):
        res = None
        if (levels is None):
            levels = int(np.floor(np.log2(self.dim))-1)
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,levels=levels,alpha=alpha,prior=type,method=variant,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from matrices import totalmatrix
        from cyt import hmc, ehmc
        wl = pywt.Wavelet(type)
        g = np.array(wl.dec_lo)
        h = np.array(wl.dec_hi)
        regx = totalmatrix(self.dim, levels, g, h)
        regy = sp.csc_matrix((1, self.dim * self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        self.Q.Lx = regx
        self.Q.b = 0.01
        self.Q.Ly = regy
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.logdensity = ltv
        self.Q.gradi = tv_grad
        if mapstart:
            x0 = np.reshape(self.map_wavelet(alpha, type=type, levels=levels, maxiter=150), (-1, 1))
            x0 = x0 + 0.000001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running " + variant.upper() + " for Besov prior (" + type + ' '  + str(levels) + ').' )
        if (variant == 'hmc'):
            solution, chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75,cmonly=retim, thinning=thinning,istep=interstep,intername=res.intermedfilename)
        else:
            solution, chain = ehmc(M, x0, self.Q, Madapt,  cmonly=retim,thinning=thinning,istep=interstep,intername=res.intermedfilename)
        #solution,chain = hmc(M, x0, self.Q, Madapt, de=0.65, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_tv(self, alpha, M=10000, Madapt=1000,mapstart=False,thinning=10,retim=True,interstep=100000):
        res = None
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tv',method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from cyt import mwg_tv as mwgt
        regN = np.diag([1] * self.dim, 0) + np.diag([-1] * (self.dim - 1), 1);
        regN = sp.csc_matrix(regN[0:-1, :])
        help = np.zeros((2, self.dim));
        help[0, 0] = 1;
        help[1, self.dim - 1] = 1
        help2 = np.hstack([np.zeros((self.dim - 2, 1)), np.eye(self.dim - 2), np.zeros((self.dim - 2, 1))])
        # regvalues = np.array([1, -1, 1])
        # offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron(sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron(sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        # combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx, regx2, regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.00
        self.Q.y = self.lines
        if (mapstart):
            x0 = np.reshape(self.map_tv(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.000001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.0 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for TV prior.")
        solution,chain= mwgt(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim,thinning=thinning,interstep=interstep,intername=res.intermedfilename)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_cauchy(self, alpha, M=10000, Madapt=1000,mapstart=False,thinning=10,retim=True,interstep=100000):
        res = None
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='cauchy',method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from cyt import mwg_cauchy as mwgc
        regN = np.diag([1]*self.dim,0) + np.diag([-1]*(self.dim-1),1);
        regN = sp.csc_matrix(regN[0:-1,:])
        help = np.zeros((2,self.dim)); help[0,0] = 1; help[1,self.dim-1] = 1
        help2 = np.hstack([np.zeros((self.dim-2,1)),np.eye(self.dim-2),np.zeros((self.dim-2,1))])
        #regvalues = np.array([1, -1, 1])
        #offsets = np.array([-self.dim + 1, 0, 1])
        # reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        # reg1d = sp.csc_matrix(reg1d)
        # reg1d[self.dim-1,self.dim-1] = 0
        # print(np.linalg.matrix_rank(reg1d.todense()))
        # regx = sp.kron(sp.eye(self.dim), reg1d)
        # regy = sp.kron(reg1d, sp.eye(self.dim))
        # regx = sp.csc_matrix(regx)
        # regy = sp.csc_matrix(regy)
        regx = sp.kron(sp.eye(self.dim), regN)
        regy = sp.kron(regN, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        regx2 = sp.kron( sp.csc_matrix(help2), sp.csc_matrix(help))
        regy2 = sp.kron( sp.csc_matrix(help), sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        #combined = sp.vstack([regy, regx], format='csc')
        combined = sp.vstack([regy, regx,regx2,regy2], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        self.Q.y = self.lines
        if mapstart:
            x0 = np.reshape(self.map_cauchy(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.000001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.0 + 0.00*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for Cauchy prior.")
        solution, chain = mwgc(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim, thinning=thinning,interstep=interstep,intername=res.intermedfilename)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_wavelet(self, alpha, M=10000, Madapt=1000,type='haar',levels=None,mapstart=False,thinning=10,retim=True,interstep=100000):
        res = None
        if (levels is None):
            levels = int(np.floor(np.log2(self.dim))-1)
        if not retim:
            res = container(crimefree=self.crimefree,totaliternum=M,levels=levels,adaptnum=Madapt,alpha=alpha,prior=type,method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,globalprefix=self.globalprefix,theta=self.theta/(2*np.pi)*360)
        from matrices import totalmatrix
        from cyt import mwg_tv as mwgt
        wl = pywt.Wavelet(type)
        g = np.array(wl.dec_lo)
        h = np.array(wl.dec_hi)
        regx = totalmatrix(self.dim, levels, g, h)
        regy = sp.csc_matrix((1, self.dim * self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        self.Q.Lx = regx
        self.Q.b = 0.0000
        self.Q.Ly = regy
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        if (mapstart):
            x0 = np.reshape(self.map_wavelet(alpha, type=type,levels=levels, maxiter=150), (-1, 1))
            x0 = x0 + 0.000001 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.0 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for Besov prior (" + type + ' '  + str(levels) + ').' )
        solution,chain= mwgt(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim,thinning=thinning,interstep=interstep,intername=res.intermedfilename)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def target(self):
        return self.targetimage

    def sinogram(self):
        plt.imshow(self.sgram, extent=[self.theta[0], self.theta[-1], -np.sqrt(2), np.sqrt(2)])
        plt.show()

    def radonww(self,image, theta_in_angles,circle=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return radon(image, theta_in_angles, circle)

    def difference(self,img):
        t = np.ravel(np.reshape(self.targetimage,(1,-1)))
        r = np.ravel(np.reshape(img,(1,-1)))
        L1 = np.linalg.norm(t-r,ord=1)/np.linalg.norm(t,ord=1)
        L2 = np.linalg.norm(t-r,ord=2)/np.linalg.norm(t,ord=2)
        return L1,L2

    def correlationrow(self,M):
        if (len(M.shape) <= 1 or M.shape[0] <= 1):
            M = M - np.mean(M)
            M = correlate(M, M, mode='full', method='fft')
            M = M[int((M.shape[0] - 1) / 2):]
            return M / M[0]

        else:
            M = M - np.mean(M, axis=1, keepdims=True)
            M = np.apply_along_axis(lambda x: correlate(x, x, mode='full', method='fft'), axis=1, arr=M)
            M = M[:, int((M.shape[1] - 1) / 2):]
            return M / np.reshape(M[:, 0], (-1, 1))

    def saveresult(self,result):
        import h5py
        filename = self.globalprefix + result.prefix + ".hdf5"
        path = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(path):
            os.makedirs(path)
        with h5py.File(filename, 'w') as f:
            for key, value in result.__dict__.items():
                if (value is None):
                    value = "None"
                if (isinstance(value, np.ndarray)):
                    compression = 'gzip'
                    value = value.astype(np.float32)
                else:
                    compression = None
                f.create_dataset(key, data=value, compression=compression)
        f.close()


if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', default="shepp.png", type=str, help='Image filename. Default=shepp.png')
    parser.add_argument('--targetsize', default=128, type=int, help='Input image is scaled to this size. Default=64')
    parser.add_argument('--crimefree', default=False, type=bool, help='Simulate sinogram with larger grid and interpolate. Default=False')
    parser.add_argument('--meas-noise', default=0.001, type=float, help='Measurement noise. Default=0.015')
    parser.add_argument('--itheta', default=180,nargs="+", type=int, help='Range and/or number of radon measurement '
    'angles in degrees. One must enter either 3 values (start angle, end angle, number of angles) or just the number of angles, in case the range 0-180 is assumed. Default=50')
    parser.add_argument('--globalprefix', default="/results/", type=str, help='Relative prefix to the script itself, if one wants to save the results. Default= /results/')
    parser.add_argument('--sampler', default="map", type=str, help='Method to use: hmc, mwg or map. Default= map')
    parser.add_argument('--levels', default=None, type=int, help='Number of DWT levels to be used. Default=None means automatic.')
    parser.add_argument('--prior', default="tikhonov", type=str,
                        help='Prior to use: tikhonov, cauchy, tv or wavelet. Default= cauchy')
    parser.add_argument('--wave', default="haar", type=str, help='DWT type to use with wavelets. Default=haar')
    parser.add_argument('--samples-num', default=200, type=int,
                        help='Number of samples to be generated within MCMC methods. Default=200, which should be completed within minutes by HMC at small dimensions.')
    parser.add_argument('--thinning', default=1, type=int,
                        help='Thinning factor for MCMC methods.  Default=1 is suitable for HMC, MwG might need thinning between 10-500. ')
    parser.add_argument('--adapt-num', default=50, type=int, help='Number of adaptations in MCMC. Default=50, which roughly suits for HMC.')
    parser.add_argument('--dataload', default=False, action='store_true', help='Use external data. Default=False')
    parser.add_argument('--alpha', default=10, type=float,
                        help='Prior alpha (regulator constant). Default=1.0, which is rather bad for all priors.')
    parser.add_argument('--omit', default=False, action='store_true',
                        help='Omit the command line arguments parsing section in the main.py')
    args = parser.parse_args()

    if len(sys.argv) > 1 and (args.omit is False) and (args.dataload is False):
        t = tomography(filename=args.file_name, targetsize=args.targetsize, itheta=args.itheta, noise=args.meas_noise,crimefree=args.crimefree,commonprefix=args.globalprefix)
        real = t.target()
        r = None
        if args.sampler == "hmc":
            if args.prior == "cauchy":
                r = t.hmcmc_cauchy(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,thinning=args.thinning)
            elif args.prior == "tv":
                r = t.hmcmc_tv(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,thinning=args.thinning)
            elif args.prior == "wavelet":
                r = t.hmcmc_wavelet(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,type=args.wave,levels=args.levels,thinning=args.thinning)
            elif args.prior == "tikhonov":
                r = t.hmcmc_tikhonov(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,thinning=args.thinning)
        elif args.sampler == "mwg":
            if args.prior == "cauchy":
                r = t.mwg_cauchy(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,thinning=args.thinning)
            elif args.prior == "tv":
                r = t.mwg_tv(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,thinning=args.thinning)
            elif args.prior == "wavelet":
                r = t.mwg_wavelet(alpha=args.alpha, M=args.samples_num, Madapt=args.adapt_num,type=args.wave,levels=args.levels,thinning=args.thinning)
        elif args.sampler == "map":
            if args.prior == "cauchy":
                r = t.map_cauchy(alpha=args.alpha,maxiter=125)
            elif args.prior == "tv":
                r = t.map_tv(alpha=args.alpha,maxiter=125)
            elif args.prior == "wavelet":
                r = t.map_wavelet(alpha=args.alpha,type=args.wave,levels=args.levels)
            elif args.prior == "tikhonov":
                r = t.map_tikhonov(alpha=args.alpha)

        plt.imshow(r)
        plt.show()

    elif  len(sys.argv) > 1 and (args.omit is False) and (args.dataload is True):
        import scipy.io
        from scipy.stats import zscore
        #m = scipy.io.loadmat('Walnut.mat')['FBP1200'].T
        #m = resize(m, (328, 328), anti_aliasing=False, preserve_range=True,order=1, mode='symmetric')
        #t = tomography(targetsize=64,commonprefix='/isot')

        #t.dataload('LotusData256.mat',"A",'LotusData256.mat','m')
        #t.dataload('CheeseData_256x180.mat', "A", 'CheeseData_256x180.mat', 'm',imsize=256)
        #t.dataload('WalnutData164.mat', "A", 'WalnutData164.mat', 'm')

        #t.lhsigmsq = 0.05
        #t.Q = argumentspack(M=t.radonoperator, y=t.lines, b=0.01, s2=0.05)
        #t.targetimage = np.random.randn(t.dim,t.dim)
        #t.theta = np.array([0,90])
        #r = t.hmcmc_tv(alpha=100, M=100, Madapt=20, thinning=1, retim=False, interstep=9, variant='ehmc')
        #r=t.map_cauchy(alpha=0.01,retim=True)


        #plt.imshow(r)
        #plt.show()



    # If we do not care the command line.
    else:

        #https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
        from collections import defaultdict
        import json
        class NestedDefaultDict(defaultdict):
            def __init__(self, *args, **kwargs):
                super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

            def __repr__(self):
                return repr(dict(self))


        '''
        angles = {'sparsestwhole': 10, 'sparsewhole': 30, 'whole': 90, 'sparsestlimited': (0, 90, 10),'sparselimited': (0, 90, 30), 'limited': (0, 90, 90)}
        noises = ( 0.015,)
        sizes = (512,)


        alphas = np.geomspace(0.1,1000,15)
        tikhoalpha = NestedDefaultDict()
        for size in sizes:
            for angletype,angle in angles.items():
                    for noise in noises:
                        bestl2 = np.Inf
                        best = 0
                        t = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                        t2 = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                        for alpha in alphas:
                            res = t.map_tikhonov(alpha, retim=False,maxiter=500)
                            res2 = t2.map_tikhonov(alpha, retim=False, maxiter=500)
                            if ((res.l2 + res2.l2)/2.0 < bestl2):
                                best = alpha
                                bestl2 = (res.l2 + res2.l2)/2.0
                        tikhoalpha[angletype][size][noise] = best

        jsontik = json.dumps(tikhoalpha)
        f = open("tikhonov.json", "w")
        f.write(jsontik)
        f.close()
        print(tikhoalpha)

        alphas = np.geomspace(0.1, 1000, 15)
        tvalpha = NestedDefaultDict()
        for size in sizes:
            for angletype, angle in angles.items():
                for noise in noises:
                    bestl2 = np.Inf
                    best = 0
                    t = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    t2 = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    for alpha in alphas:
                        res = t.map_tv(alpha, retim=False, maxiter=500)
                        res2 = t2.map_tv(alpha, retim=False, maxiter=500)
                        if ((res.l2 + res2.l2) / 2.0 < bestl2):
                            best = alpha
                            bestl2 = (res.l2 + res2.l2) / 2.0
                    tvalpha[angletype][size][noise] = best

        jsontv = json.dumps(tvalpha)
        f = open("tv.json", "w")
        f.write(jsontv)
        f.close()
        print(tvalpha)

        alphas = np.geomspace(0.000001, 5, 15)
        cauchyalpha = NestedDefaultDict()
        for size in sizes:
            for angletype, angle in angles.items():
                for noise in noises:
                    bestl2 = np.Inf
                    best = 0
                    t = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    t2 = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    for alpha in alphas:
                        res = t.map_cauchy(alpha, retim=False, maxiter=500)
                        res2 = t2.map_cauchy(alpha, retim=False, maxiter=500)
                        if ((res.l2 + res2.l2) / 2.0 < bestl2):
                            best = alpha
                            bestl2 = (res.l2 + res2.l2) / 2.0
                    cauchyalpha[angletype][size][noise] = best

        jsoncau= json.dumps(cauchyalpha)
        f = open("cauchy.json", "w")
        f.write(jsoncau)
        f.close()
        print(cauchyalpha)

        alphas = np.geomspace(0.01, 1000, 15)
        haaralpha = NestedDefaultDict()
        for size in sizes:
            for angletype, angle in angles.items():
                for noise in noises:
                    bestl2 = np.Inf
                    best = 0
                    t = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    t2 = tomography("slices/299.mat", size, angle, noise, crimefree=True, commonprefix='/results/')
                    for alpha in alphas:
                        res = t.map_wavelet(alpha, type='haar', retim=False, maxiter=500)
                        res2 = t2.map_wavelet(alpha, type='haar', retim=False, maxiter=500)
                        if ((res.l2 + res2.l2) / 2.0 < bestl2):
                            best = alpha
                            bestl2 = (res.l2 + res2.l2) / 2.0
                    haaralpha[angletype][size][noise] = best

        jsonhaar = json.dumps(haaralpha)
        f = open("haar.json", "w")
        f.write(jsonhaar)
        f.close()
        print(haaralpha)
        exit(0)
        '''
        np.random.seed(1)
        t = tomography("koe.png", 64, 64, 0.02, crimefree=False)
        res = t.map_cauchy(0.1, retim=True,isotropic=True)
        plt.imshow(res)
        plt.show()

        exit(0)

        tikhoalpha = {"sparsestwhole": {512: {0.015: 10.0}}, "sparsewhole": {512: {0.015: 10.0}}, "whole": {512: {0.015: 10.0}}, "sparsestlimited": {512: {0.015: 0.372759372031494}}, "sparselimited": {512: {0.015: 0.7196856730011519}}, "limited": {512: {0.015: 2.6826957952797246}}}
        tvalpha = {"sparsestwhole": {512: {0.015: 0.7196856730011519}}, "sparsewhole": {512: {0.015: 1.3894954943731375}}, "whole": {512: {0.015: 2.6826957952797246}}, "sparsestlimited": {512: {0.015: 0.1}}, "sparselimited": {512: {0.015: 0.372759372031494}}, "limited": {512: {0.015: 0.372759372031494}}}
        haaralpha = {"sparsestwhole": {512: {0.015: 1.3894954943731375}}, "sparsewhole": {512: {0.015: 1.3894954943731375}}, "whole": {512: {0.015: 3.1622776601683795}}, "sparsestlimited": {512: {0.015: 0.2682695795279726}}, "sparselimited": {512: {0.015: 1.3894954943731375}}, "limited": {512: {0.015: 1.3894954943731375}}}
        cauchyalpha = {"sparsestwhole": {512: {0.015: 0.0014142135623730952}}, "sparsewhole": {512: {0.015: 0.003986470631277378}}, "whole": {512: {0.015: 0.031676392175331615}}, "sparsestlimited": {512: {0.015: 2.0}}, "sparselimited": {512: {0.015: 0.7095065752033103}}, "limited": {512: {0.015: 0.25169979012836524}}}

        noises = (0.015,)
        sizes = (512,)
        angles = {'sparsewhole': 30, 'whole': 90}
		
		
        for slice in range(120,121):
            for size in sizes:
                for angletype,theta in angles.items():
                    for noise in noises:
                        t = tomography("slices/" + str(slice) + ".mat", size, theta, noise, crimefree=True, commonprefix='/results/')


                        res = t.map_tikhonov(tikhoalpha[angletype][size][noise], order=1, retim=False)
                        t.saveresult(res)

                        res = t.map_tv(tvalpha[angletype][size][noise], retim=False)
                        t.saveresult(res)

                        res = t.map_cauchy(cauchyalpha[angletype][size][noise], retim=False)
                        t.saveresult(res)

                        res = t.map_wavelet(haaralpha[angletype][size][noise], type='haar', retim=False)
                        t.saveresult(res)

                        res = t.mwg_tv(tvalpha[angletype][size][noise], mapstart=True, M=900000, Madapt=500000,
                                       retim=False, thinning=300,interstep=100000)
                        t.saveresult(res)

                        res = t.mwg_cauchy(cauchyalpha[angletype][size][noise], mapstart=True, M=900000, Madapt=500000,
                                           retim=False, thinning=300,interstep=100000)
                        t.saveresult(res)

                        res = t.mwg_wavelet(haaralpha[angletype][size][noise], mapstart=True, type='haar', M=900000,
                                           Madapt=500000, retim=False, thinning=300,interstep=100000)
                        t.saveresult(res)
						
                        res = t.hmcmc_tv(tvalpha[angletype][size][noise], mapstart=True, M=4100, Madapt=100, retim=False,
                                             thinning=1,interstep=100)
                        t.saveresult(res)

                        res = t.hmcmc_cauchy(cauchyalpha[angletype][size][noise], mapstart=True, M=4100, Madapt=100,
                                             retim=False, thinning=1,interstep=100)
                        t.saveresult(res)

                        res = t.hmcmc_wavelet(haaralpha[angletype][size][noise], mapstart=True, M=4100, Madapt=100,
                                             retim=False, thinning=1,interstep=100)
                        t.saveresult(res)


                        



                        



