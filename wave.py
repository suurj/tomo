import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import  pywt
from matrices import waveletonce,totalmatrix
from skimage.transform import rescale
import scipy.sparse as sp
from scipy.sparse import csr_matrix,csc_matrix,lil_matrix, coo_matrix, dok_matrix

# def totalmatrix(n,levels,g,h):
#     if (levels<1 or np.mod(n,2**levels) != 0 ):
#         raise Exception('DWT level mismatch.')
#
#     Gs = []
#     Hs = []
#     Hprev = []
#     for i in range(0,levels):
#         Gi, Hi = waveletonce(g, h, int(n/2**(i)))
#         Gs.append(Gi)
#         Hs.append(Hi)
#         if (len(Hprev) == 0):
#             Hprev.append(Hi)
#         else:
#             Hprev.append(Hi.dot(Hprev[i-1]))
#
#     for i in range(0,levels-1):
#         if (i == 0):
#             M = sp.kron(G[i],G[i])
#             M = sp.vstack((sp.kron(G,H),M))
#             M = sp.vstack((sp.kron(H, G), M))
#         else:
#             p = Hprev[i-1]
#             gp = Gs[i].dot(p)
#             hp = Hs[i].dot(p)
#             M = sp.vstack((sp.kron(gp,gp),M))
#             M = sp.vstack((sp.kron(gp, hp), M))
#             M = sp.vstack((sp.kron(hp, gp), M))
#
#     if (levels ==1):
#         gp = Gs[0]
#         hp = Hs[0]
#         M = sp.kron(gp, gp)
#         M = sp.vstack((sp.kron(gp, hp), M))
#         M = sp.vstack((sp.kron(hp, gp), M))
#         M = sp.vstack((sp.kron(hp, hp), M))
#     else:
#         p = Hprev[levels - 2]
#         gp = Gs[levels-1].dot(p)
#         hp = Hs[levels-1].dot(p)
#         M = sp.vstack((sp.kron(gp, gp), M))
#         M = sp.vstack((sp.kron(gp, hp), M))
#         M = sp.vstack((sp.kron(hp, gp), M))
#         M = sp.vstack((sp.kron(hp, hp), M))
#
#     return  csc_matrix(M)


type = 'db5'
im = imread("shepp128.png", as_gray=True)
im = rescale(im, scale=1, mode='edge', multichannel=False)
n = im.shape[0]
im = np.array(im)
wl = pywt.Wavelet(type)
g = np.array(wl.dec_lo)
h = np.array(wl.dec_hi)
(G,H) = waveletonce(g,h,n)
G = csc_matrix(G)
H = csc_matrix(H)

imfl = np.reshape(im,(-1,1))
M=totalmatrix(n,1,g,h)
#M = sp.kron(G,H,format='csc')
wlist=pywt.wavedec2(im, type, mode='periodization', level=1)
q = M.dot(imfl)
q = np.reshape(q[0:int((n/2)**2), 0],(int(n/2),int(n/2)))




# t = np.linspace(0,4*np.pi,n,True)
# sig = np.sin(t) + 0.1*np.sin(10*t)
# y = G@sig

# (cA, cD) = pywt.dwt(sig,mode='periodization', wavelet=type)
# tot = np.hstack((cA,cD))
plt.imshow(wlist[0])

plt.figure()
plt.imshow(q)
plt.show()