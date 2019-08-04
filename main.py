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
from cyt import tfun_cauchy as lcauchy, tfun_tikhonov as ltikhonov, tikhonov_grad, tfun_tv as ltv, tv_grad, cauchy_grad, \
    argumentspack

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

    def __init__(self, filename, targetsize=128, itheta=50, noise=0.0,  commonprefix="", dimbig = 607, N_thetabig=421, crimefree=False,lhdev=None):
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
        if targetsize > 512:
            raise Exception(
                'Dimensions of the target image are too large (' + str(targetsize) + 'x' + str(targetsize) + ')')

        img = imread(filename, as_gray=True)
        (dy, dx) = img.shape
        if (dy != dx):
            raise Exception('Image is not rectangular.')
        
        self.targetimage = imread(filename, as_gray=True)
        self.targetimage = resize(self.targetimage, (self.dim, self.dim), anti_aliasing=False, preserve_range=True,
                            order=1, mode='symmetric')
        if self.crimefree:
            image = imread(filename, as_gray=True)
            image = resize(image, (self.dimbig, self.dimbig), anti_aliasing=False, preserve_range=True,
                                order=1, mode='symmetric')
            (self.simsize, _) = image.shape

        else:
            image = imread(filename, as_gray=True)
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
            from matrices import radonmatrix

            self.radonoperator = radonmatrix(self.dim, self.theta)
            sp.save_npz(fname, self.radonoperator)

        # In the case of inverse-crime free tomography,
        # one might use the Radon tool from scikit-image
        # or construct another Radon matrix and calculate a sinogram with that. The former is definitely faster and also preferred,
        # since different methods are used to simulate and reconcstruct the image.

        #if crimefree and (not os.path.isfile(fnamebig)):
        #    from matrices import radonmatrix

        #    self.radonoperatorbig = radonmatrix(self.dimbig, self.thetabig)
        #    sp.save_npz(fnamebig, self.radonoperatorbig)

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
        self.method = 'L-BFGS-B'

    def mincb(self,_):
        self.pbar.update(1)

    def map_tikhonov(self, alpha=1.0, order=1,maxiter=400,retim=True):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree, prior='tikhonov', levels=order, method='map', noise=self.noise, imagefilename=self.filename,
                            target=self.targetimage, targetsize=self.dim, theta=self.theta)
        if (order == 2):
            regvalues = np.array([2, -1, -1, -1, -1])
            offsets = np.array([0, 1, -1, self.dim - 1, -self.dim + 1])
            reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        else:
            regvalues = np.array([1, -1, 1])
            offsets = np.array([-self.dim + 1, 0, 1])
            reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        alpha = alpha
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        combined = sp.vstack([regy, regx], format='csc')
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
            res.finish(solution, self.difference(solution), iters=iters)
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
                            target=self.targetimage, targetsize=self.dim, theta=self.theta/(2*np.pi)*360)
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
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
            res.finish(solution, self.difference(solution), iters=iters)
            return res
        else:
            return solution

    def tfun_tv(self, x):
        return -ltv(x, self.Q)

    def grad_tv(self, x):
        x = x.reshape((-1, 1))
        q = -tv_grad(x, self.Q)
        return np.ravel(q)

    def map_cauchy(self, alpha=0.05, maxiter=400,retim=True):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree,prior='cauchy',method='map',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        print("Running MAP estimate for Cauchy prior.")
        self.pbar = tqdm(total=np.Inf,file=sys.stdout)
        solution = minimize(self.tfun_cauchy, x0, method=self.method, jac=self.grad_cauchy,
                            options={'maxiter': maxiter, 'disp': False},callback=self.mincb)
        self.pbar.close()
        iters = solution.nit
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(solution,self.difference(solution),iters=iters)
            return res
        else:
            return solution

    def tfun_cauchy(self, x):
        return -lcauchy(x, self.Q)

    def grad_cauchy(self, x):
        x = x.reshape((-1, 1))
        ans = -cauchy_grad(x, self.Q)
        return (np.ravel(ans))

    def map_wavelet(self, alpha=1.0, type='haar', maxiter=400,levels=3 ,retim=True):
        res = None
        if not retim:
            res = container(alpha=alpha,crimefree=self.crimefree,prior=type,method='map',levels=levels,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
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
        print("Running MAP estimate for Besov prior.")
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
            res.finish(solution, self.difference(solution), iters=iters)
            return res
        else:
            return solution

    def hmcmc_tikhonov(self, alpha, M=100, Madapt=20, order=1,mapstart=False,thinning=1,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tikhonov',method='hmc',levels=order,noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc
        if (order == 2):
            regvalues = np.array([2, -1, -1, -1, -1])
            offsets = np.array([0, 1, -1, self.dim - 1, -self.dim + 1])
            reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        else:
            regvalues = np.array([1, -1, 1])
            offsets = np.array([-self.dim + 1, 0, 1])
            reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        self.Q.logdensity = ltikhonov
        self.Q.gradi = tikhonov_grad
        self.Q.y = self.lines
        if (mapstart):
            x0 = np.reshape(self.map_tikhonov(alpha,maxiter=150),(-1,1))
            x0 = x0 + 0.01*np.random.rand(self.dim*self.dim,1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        solution,chain = hmc(M, x0, self.Q, Madapt, de=0.651, gamma=0.05, t0=10.0, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_tv(self, alpha, M=100, Madapt=20,mapstart=False,thinning=1,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tv',method='hmc',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        self.Q.logdensity = ltv
        self.Q.gradi = tv_grad
        if mapstart:
            x0 = np.reshape(self.map_tv(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running HMC for TV prior.")
        solution,chain = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_cauchy(self, alpha, M=100, Madapt=20,thinning=1,mapstart=True,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='cauchy',method='hmc',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from cyt import hmc
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
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
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running HMC for Cauchy prior.")
        solution,chain = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cmonly=retim, thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def hmcmc_wavelet(self, alpha, M=100, Madapt=20, type='haar',levels=3,mapstart=False,thinning=1,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,levels=levels,alpha=alpha,prior=type,method='hmc',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from matrices import totalmatrix
        from cyt import hmc
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
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running HMC for Besov prior.")
        solution,chain = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_tv(self, alpha, M=10000, Madapt=1000,mapstart=False,thinning=10,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='tv',method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from cyt import mwg_tv as mwgt
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.00
        self.Q.y = self.lines
        if (mapstart):
            x0 = np.reshape(self.map_tv(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for TV prior.")
        solution,chain= mwgt(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim,thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_cauchy(self, alpha, M=10000, Madapt=1000,mapstart=False,thinning=10,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,adaptnum=Madapt,alpha=alpha,prior='cauchy',method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
        from cyt import mwg_cauchy as mwgc
        regvalues = np.array([1, -1, 1])
        offsets = np.array([-self.dim + 1, 0, 1])
        reg1d = sp.diags(regvalues, offsets, shape=(self.dim, self.dim))
        regx = sp.kron(sp.eye(self.dim), reg1d)
        regy = sp.kron(reg1d, sp.eye(self.dim))
        regx = sp.csc_matrix(regx)
        regy = sp.csc_matrix(regy)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        alpha = alpha
        combined = sp.vstack([regy, regx], format='csc')
        empty = sp.csc_matrix((1, self.dim * self.dim))
        self.Q.Lx = combined
        self.Q.Ly = empty
        self.Q.a = alpha
        self.Q.s2 = self.lhsigmsq
        self.Q.b = 0.01
        self.Q.y = self.lines
        if mapstart:
            x0 = np.reshape(self.map_cauchy(alpha, maxiter=150), (-1, 1))
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for Cauchy prior.")
        solution, chain = mwgc(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim, thinning=thinning)
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        if not retim:
            res.finish(result=solution, error=self.difference(solution),chain=chain,thinning=thinning)
            return res
        else:
            return solution

    def mwg_wavelet(self, alpha, M=10000, Madapt=1000,type='haar',levels=3,mapstart=False,thinning=10,retim=True):
        res = None
        if not retim:
            res = container(totaliternum=M,levels=levels,adaptnum=Madapt,alpha=alpha,prior=type,method='mwg',noise=self.noise,imagefilename=self.filename,target=self.targetimage,targetsize=self.dim,theta=self.theta/(2*np.pi)*360)
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
            x0 = x0 + 0.01 * np.random.rand(self.dim * self.dim, 1)
        else:
            x0 = 0.2 + 0.01*np.random.randn(self.dim * self.dim, 1)
        print("Running MwG MCMC for Besov prior.")
        solution,chain= mwgt(M, Madapt, self.Q, x0, sampsigma=1.0, cmonly=retim,thinning=thinning)
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
                    compression = 'lzf'
                else:
                    compression = None
                f.create_dataset(key, data=value, compression=compression)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', default="shepp.png", type=str, help='Image filename. Default=shepp.png')
    parser.add_argument('--targetsize', default=64, type=int, help='Input image is scaled to this size. Default=64')
    parser.add_argument('--crimefree', default=False, type=bool, help='Simulate sinogram with larger grid and interpolate. Default=False')
    parser.add_argument('--meas-noise', default=0.01, type=float, help='Measurement noise. Default=0.01')
    parser.add_argument('--itheta', default=50,nargs="+", type=int, help='Range and/or number of radon measurement '
    'angles in degrees. One must enter either 3 values (start angle, end angle, number of angles) or just the number of angles, in case the range 0-180 is assumed. Default=50')
    parser.add_argument('--globalprefix', default="/results/", type=str, help='Relative prefix to the script itself, if one wants to save the results. Default= /results/')
    parser.add_argument('--sampler', default="map", type=str, help='Method to use: hmc, mwg or map. Default= map')
    parser.add_argument('--levels', default=2, type=int, help='Number of DWT levels to be used. Default= 2')
    parser.add_argument('--prior', default="cauchy", type=str,
                        help='Prior to use: tikhonov, cauchy, tv or wavelet. Default= cauchy')
    parser.add_argument('--wave', default="haar", type=str, help='DWT type to use with wavelets. Default=haar')
    parser.add_argument('--samples-num', default=200, type=int,
                        help='Number of samples to be generated within MCMC methods. Default=200, which should be completed within minutes even by HMC.')
    parser.add_argument('--thinning', default=1, type=int,
                        help='Thinning factor for MCMC methods.  Default=1 is suitable for HMC, MwG might need thinning between 10-100. ')
    parser.add_argument('--adapt-num', default=50, type=int, help='Number of adaptations in MCMC. Default=50, which roughly suits for HMC.')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Prior alpha (regulator constant). Default=1.0, which is rather bad for all priors.')
    parser.add_argument('--omit', default=False, type=bool,
                        help='Omit the command line arguments parsing section in the main.py')
    args = parser.parse_args()

    if len(sys.argv) > 1 and (args.omit is False):
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
                r = t.map_cauchy(alpha=args.alpha)
            elif args.prior == "tv":
                r = t.map_tv(alpha=args.alpha)
            elif args.prior == "wavelet":
                r = t.map_wavelet(alpha=args.alpha,type=args.wave,levels=args.levels)
            elif args.prior == "tikhonov":
                r = t.map_tikhonov(alpha=args.alpha)

        plt.imshow(r)
        plt.show()

    # If we do not care the command line
    else:
        np.random.seed(3)
        #theta = (0, 90, 50)
        theta = 50
        t = tomography("shepp.png", 64, theta, 0.05, crimefree=False,commonprefix='/results/')
        real = t.target()
        # t.saveresult(real)
        # sg = t.sinogram()
        #t.sinogram()

        # t.normalizedsgram = t.radonww()
        #t.sinogram()

        # sg2 = t.radonww()
        # t = tomography("shepp.png",0.1,20,0.2)
        # r = t.mwg_cauchy(0.01,5000,200)
        # r = t.hmcmc_tv(5,250,30)
        # r = t.hmcmc_cauchy(0.1,150,30)
        # r = t.hmcmc_tikhonov(50, 200, 20)
        # #r = t.hmcmc_wavelet(25, 250, 20,type='haar')
        # #print(np.linalg.norm(real - r))
        # # tt = time.time()
        # #
        #r = t.map_wavelet(5)
        res = t.mwg_cauchy(0.01, 200, 100, thinning=10, mapstart=False, retim=False)
        #res = t.hmcmc_cauchy(0.01, 230, 30, thinning=1, mapstart=True, retim=False)
        t.saveresult(res)
        #
        r = res.result
        # plt.plot(res.chain[5656,:])
        # plt.figure()
        print(t.difference(r))
        # r = t.map_cauchy(0.01,True)

        # r = t.map_tikhonov(10,True,order=1)
        # #
        # # # # print(time.time()-tt)
        # # # #
        # r = t.hmcmc_cauchy(0.01,100,20)
        # r = t.mwg_cauchy(0.01, 1000, 100)
        # print(time.time()-tt)
        # r = t.hmcmc_tv(10, 200, 20)
        # r = t.hmcmc_cauchy(100/(t.dim**2), 250, 30)
        plt.imshow(r)
        # # # #plt.plot(r[3000,:],r[2000,:],'*r')
        plt.clim(0, 1)
        plt.figure()
        #r2 = t.hmcmc_cauchy(0.001, 200, 20)
        r2 = t.map_cauchy(0.01)
        #r2 = t.map_cauchy(0.001)
        #r2 = t.mwg_wavelet(10,5000,2000,levels=6,mapstart=True)
        #r2 = t.mwg_tv( 5,2000,200)
        #r2 = t.map_tv(5)
        # # #print(np.linalg.norm(real - r))
        # # #q = iradon_sart(q, theta=theta)
        # # #r2 = t.map_tikhonov(50.0)
        # # #tt = time.time()
        # r2 = t.map_tikhonov(1)
        # r2 = t.map_wavelet(0.5,'db2')
        print(t.difference(r2))
        # # # #print(time.time()-tt)
        plt.imshow(r2)
        plt.clim(0, 1)
        plt.show()

#
#
