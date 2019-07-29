from skimage.io import imread
from skimage.transform import radon, rescale, resize
import warnings
import numpy as np
import pywt
import scipy.interpolate as interpolate
from scipy.optimize import minimize
import time
import math
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
from cyt import tfun_cauchy as lcauchy, tfun_tikhonov as ltikhonov, tikhonov_grad, tfun_tv as ltv, tv_grad, cauchy_grad, \
    argumentspack


class tomography:

    def __init__(self, filename, targetsize=128, itheta=40, noise=0.0, crimefree=False, globalprefix=""):
        self.globalprefix = globalprefix
        self.dim = targetsize
        fnamebig = None
        self.thetabig = None
        self.dimbig = 128
        self.N_thetabig = 100
        self.N_rbig = math.ceil(np.sqrt(2) * self.dimbig)
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
        if crimefree:
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

        if isinstance(itheta, (int, np.int32, np.int64)):
            self.theta = np.linspace(0., 180., itheta, endpoint=False)
            self.theta = self.theta / 360 * 2 * np.pi
            (self.N_r, self.N_theta) = (math.ceil(np.sqrt(2) * self.dim), itheta)
            self.rhoo = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_r, endpoint=True)
            fname = 'radonmatrix/full-{0}x{1}.npz'.format(str(self.dim), str(self.N_theta))

            if crimefree:
                self.thetabig = np.linspace(0, 180, self.N_thetabig, endpoint=False)
                self.thetabig = self.thetabig / 360 * 2 * np.pi
                self.rhoobig = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_rbig, endpoint=True)
                fnamebig = 'radonmatrix/full-{0}x{1}.npz'.format(str(self.dimbig), str(self.N_thetabig))

        else:
            self.theta = np.linspace(itheta[0], itheta[1], itheta[2], endpoint=False)
            self.theta = self.theta / 360 * 2 * np.pi
            (self.N_r, self.N_theta) = (
                math.ceil(np.sqrt(2) * targetsize), itheta[2])
            self.rhoo = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_r, endpoint=True)
            fname = 'radonmatrix/{0}_{1}-{2}x{3}.npz'.format(str(itheta[0]), str(itheta[1]), str(self.dim), str(self.N_theta))

            if (crimefree):
                self.thetabig = np.linspace(itheta[0], itheta[1], self.N_thetabig, endpoint=False)
                self.thetabig = self.thetabig / 360 * 2 * np.pi
                self.rhoobig = np.linspace(np.sqrt(2), -np.sqrt(2), self.N_rbig, endpoint=True)
                fnamebig = 'radonmatrix/{0}_{1}-{2}x{3}.npz'.format(str(itheta[0]), str(itheta[1]), str(self.dimbig),str(self.N_thetabig))

        if not os.path.isfile(fname):
            from matrices import radonmatrix

            self.radonoperator = radonmatrix(self.dim, self.theta)
            sp.save_npz(fname, self.radonoperator)

        if crimefree and (not os.path.isfile(fnamebig)):
            from matrices import radonmatrix

            self.radonoperatorbig = radonmatrix(self.dimbig, self.thetabig)
            sp.save_npz(fnamebig, self.radonoperatorbig)

        self.radonoperator = sp.load_npz(fname)
        self.radonoperator = sp.csc_matrix(self.radonoperator)
        self.radonoperator = self.radonoperator / self.dim

        if (crimefree):
            self.radonoperatorbig = sp.load_npz(fnamebig) / self.dimbig
            simulated = self.radonoperatorbig @ self.flattened
            maxvalue = np.max(simulated)
            simulated = simulated + maxvalue * noise * np.random.randn(self.N_rbig * self.N_thetabig, 1)
            self.sgramsim = np.reshape(simulated, (self.N_rbig, self.N_thetabig))
            interp = interpolate.RectBivariateSpline(-self.rhoobig, self.thetabig, self.sgramsim)
            self.sgram = interp(-self.rhoo, self.theta)
            self.lines = np.reshape(self.sgram, (-1, 1))

        else:
            simulated = self.radonoperator @ self.flattened
            maxvalue = np.max(simulated)
            noiserealization = np.random.randn(self.N_r * self.N_theta, 1)
            self.lines = simulated + maxvalue * noise * noiserealization
            self.sgram = np.reshape(self.lines, (self.N_r, self.N_theta))

        if noise == 0:
            noise = 0.01
        self.lhsigmsq = (maxvalue * noise) ** 2
        self.Q = argumentspack(M=self.radonoperator, y=self.lines, b=0.01, s2=self.lhsigmsq)

    def map_tikhonov(self, alpha=1.0, display=False, order=1):
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

        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        solution = minimize(self.tfun_tikhonov, x0, method='L-BFGS-B', jac=self.grad_tikhonov,
                            options={'maxiter': 430, 'disp': display})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        return solution

    def tfun_tikhonov(self, x):
        return -ltikhonov(x, self.Q)

    def grad_tikhonov(self, x):
        x = x.reshape((-1, 1))
        ans = -tikhonov_grad(x, self.Q)
        return np.ravel(ans)

    def map_tv(self, alpha=1.0, display=False):
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
        solution = minimize(self.tfun_tv, x0, method='L-BFGS-B', jac=self.grad_tv,
                            options={'maxiter': 430, 'disp': display})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        return solution

    def tfun_tv(self, x):
        return -ltv(x, self.Q)

    def grad_tv(self, x):
        x = x.reshape((-1, 1))
        q = -tv_grad(x, self.Q)
        return np.ravel(q)

    def map_cauchy(self, alpha=1.0, display=False):
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
        # L-BFGS-B
        solution = minimize(self.tfun_cauchy, x0, method='L-BFGS-B', jac=self.grad_cauchy,
                            options={'maxiter': 450, 'disp': display})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))

        return solution

    def tfun_cauchy(self, x):
        return -lcauchy(x, self.Q)

    def grad_cauchy(self, x):
        x = x.reshape((-1, 1))
        ans = -cauchy_grad(x, self.Q)
        return (np.ravel(ans))

    def map_wavelet(self, alpha=1.0, type='haar', display=False):
        from matrices import totalmatrix
        wl = pywt.Wavelet(type)
        g = np.array(wl.dec_lo)
        h = np.array(wl.dec_hi)
        regx = totalmatrix(self.dim, 6, g, h)
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

        x0 = 1 + 0.05 * np.random.randn(self.dim * self.dim, )
        solution = minimize(self.tfun_tv, x0, method='L-BFGS-B', jac=self.grad_tv,
                            options={'maxiter': 230, 'disp': display})
        solution = solution.x
        solution = np.reshape(solution, (-1, 1))
        solution = np.reshape(solution, (self.dim, self.dim))
        return solution

    def hmcmc_tikhonov(self, alpha, M=100, Madapt=20, order=1):
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
        # x0 = np.reshape(self.map_tikhonov(alpha),(-1,1))
        # x0 = x0 + 1*np.random.rand(self.dim*self.dim,1)
        x0 = 0.2 * np.ones((self.dim * self.dim, 1))
        cm = hmc(M, x0, self.Q, Madapt, de=0.651, gamma=0.05, t0=10.0, kappa=0.75, cm=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def mwg_tv(self, alpha, M=10000, Madapt=1000):
        from cyt import mwg_tv
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
        # x0 = np.reshape(self.map_tikhonov(alpha),(-1,1))
        # x0 = x0 + 1*np.random.rand(self.dim*self.dim,1)
        x0 = 0.2 * np.ones((self.dim * self.dim, 1))
        cm = mwg_tv(M, Madapt, self.Q, x0, sampsigma=1.0, cmesti=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def mwg_cauchy(self, alpha, M=10000, Madapt=1000):
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
        # print(self.radonoperator.shape)
        # print(regx.shape)
        # x0 = np.reshape(self.map_tikhonov(alpha),(-1,1))
        # x0 = x0 + 1*np.random.rand(self.dim*self.dim,1)
        x0 = 0.5 * np.ones((self.dim * self.dim, 1))
        cm = mwgc(M, Madapt, self.Q, x0, sampsigma=1.0, cmesti=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def hmcmc_tv(self, alpha, M=100, Madapt=20):
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
        x0 = 0.2 * np.ones((self.dim * self.dim, 1))
        cm = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, kappa=0.75, cm=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def hmcmc_cauchy(self, alpha, M=100, Madapt=20):
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
        x0 = 0.5 * np.ones((self.dim * self.dim, 1))
        cm = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cm=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def hmcmc_wavelet(self, alpha, M=100, Madapt=20, type='haar'):
        from matrices import totalmatrix
        from cyt import hmc
        wl = pywt.Wavelet(type)
        g = np.array(wl.dec_lo)
        h = np.array(wl.dec_hi)
        regx = totalmatrix(self.dim, 6, g, h)
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
        x0 = 0.5 * np.ones((self.dim * self.dim, 1))
        cm = hmc(M, x0, self.Q, Madapt, de=0.6, gamma=0.05, t0=10.0, epsilonwanted=None, kappa=0.75, cm=True)
        cm = np.reshape(cm, (-1, 1))
        cm = np.reshape(cm, (self.dim, self.dim))
        return cm

    def target(self):
        return self.targetimage

    def sinogram(self):
        plt.imshow(self.sgram, extent=[self.theta[0], self.theta[-1], -np.sqrt(2), np.sqrt(2)])
        plt.show()

    def radonww(self, circle=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return radon(self.targetimage, self.theta / (2 * np.pi) * 360, circle)

    def saveresult(self, img, prefix=""):
        name = time.strftime(self.globalprefix + prefix + "%d-%m-%Y_%H%M%S.npy")
        np.save(name, img)


if __name__ == "__main__":
    np.random.seed(3)
    theta = (0, 90, 50)
    #theta = 20
    t = tomography("shepp.png", 64, theta, 0.05, crimefree=False)
    real = t.target()
    # t.saveresult(real)
    # sg = t.sinogram()
    # t.sinogram()

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
    # r = t.map_tv(1,True)
    # r = t.map_cauchy(0.01,True)
    # r = t.map_tikhonov(10 / (t.dim*t.dim))
    # r = t.map_tikhonov(10,True,order=1)
    # #
    # # # # print(time.time()-tt)
    # # # #
    # tt = time.time()
    # r = t.hmcmc_cauchy(0.01,100,20)
    # r = t.mwg_cauchy(0.01, 1000, 100)
    # print(time.time()-tt)
    # r = t.hmcmc_tv(10, 200, 20)
    # r = t.hmcmc_cauchy(100/(t.dim**2), 250, 30)
    # plt.imshow(r)
    # # # #plt.plot(r[3000,:],r[2000,:],'*r')
    # plt.clim(0, 1)
    # plt.figure()
    #r2 = t.hmcmc_cauchy(0.001, 200, 20)
    #r2 = t.map_tv(5)
    #r2 = t.map_cauchy(0.001)
    #r2 = t.map_cauchy(0.01)
    r2 = t.map_cauchy( 0.01)
    #r2 = t.map_tv(5)
    # # #print(np.linalg.norm(real - r))
    # # #q = iradon_sart(q, theta=theta)
    # # #r2 = t.map_tikhonov(50.0)
    # # #tt = time.time()
    # r2 = t.map_tikhonov(1)
    # r2 = t.map_wavelet(0.5,'db2')
    #print(np.linalg.norm(np.reshape(real - r, (-1, )),ord=2))
    print(np.linalg.norm(np.reshape(real - r2, (-1, )), ord=2))
    # # # #print(time.time()-tt)
    plt.imshow(r2)
    plt.clim(0, 1)
    plt.show()

#
#
