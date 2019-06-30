import numpy as np
import matplotlib.pyplot as plt
import  pywt
from wavelet import waveletonce

type = 'bior1.5'
n = 16
wl = pywt.Wavelet(type)
g = np.array(wl.dec_lo)
h = np.array(wl.dec_hi)
M = waveletonce(g,h,n).toarray()
#
# nhalf = int(n/2)
# g = np.ravel(g)
# h = np.ravel(h)
# fl = g.shape[0]
# fg = np.flip(g,0)
# fh = np.flip(h,0)
# H1 = np.zeros((nhalf,n))
# G1 = np.zeros((nhalf,n))
#
# for row in range(0, nhalf):
#     for col in range(0, fl):
#         c = int(np.mod(-fl / 2 + col +1 + (row) * 2, n))
#         H1[row, c] = H1[row, c] + fg[col];
#         G1[row, c] = G1[row, c] + fh[col];
#
# M=np.vstack((H1, G1))


t = np.linspace(0,4*np.pi,n,True)
sig = np.sin(t) + 0.1*np.sin(10*t)
y = M@sig

(cA, cD) = pywt.dwt(sig,mode='periodization', wavelet=type)
tot = np.hstack((cA,cD))
plt.stem(tot)

plt.figure()
plt.stem(y)

plt.show()