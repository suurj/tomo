import pandas as pd
import h5py
import os
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interpolate
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import iradon_sart
import scipy.io
from matplotlib import rc
from scipy.signal import correlate
import  sqlite3
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from matplotlib.ticker import LogFormatter
from matplotlib import font_manager as fm, rcParams
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.ticker import LogFormatterExponent
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm, colors as mcolors


noiselevels = [1.5]
angles = [10,30,90]
maxangles = [90,180]
targetsizes = [512]


def correlationrow( M):
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



dire = '/home/user/tomo/tv_hmc_30/'
os.chdir(dire)


def talleta(img,fname,var=False):
    #rc('text', usetex=True)
    fig, ax = plt.subplots()
    #plt.rcParams['text.latex.unicode'] = True

    colors = [(256 / 256, 256 / 256, 256 / 256), (232 / 256, 201 / 256, 163 / 256)]
    map1 = LinearSegmentedColormap.from_list('Map1', colors, N=256)
    plt.register_cmap(cmap=map1)

    colors2 = [(232 / 256, 201 / 256, 163 / 256), (122 / 256, 112 / 256, 87 / 256)]
    map2 = LinearSegmentedColormap.from_list('Map2', colors2, N=256)
    plt.register_cmap(cmap=map2)

    colors3 = [(122 / 256, 112 / 256, 87 / 256), (93 / 256, 93 / 256, 107 / 256)]
    map3 = LinearSegmentedColormap.from_list('Map3', colors3, N=256)
    plt.register_cmap(cmap=map3)

    ilma = cm.get_cmap('Map1', 256)
    ilmavarit = ilma(np.linspace(0, 1, 256 * 1))

    puu = cm.get_cmap('Map2', 256)
    puuvarit = puu(np.linspace(0, 1, 256 * 1))

    metalli = cm.get_cmap('Map3', 256)
    metallivarit = metalli(np.linspace(0, 1, 256 * 8))

    newcolors = np.vstack((ilmavarit, puuvarit, metallivarit))

    map4 = ListedColormap(newcolors, name='Map4')
    plt.register_cmap(cmap=map4)

    plt.rcParams['image.cmap'] = 'Map4'


    if (var == False):
        plt.imshow(img)
        plt.clim(0, 10)
        plt.axis('off')
        # # ax.set_position([0, 0, 1, 1])
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        rc('text', usetex=True)
        plt.tight_layout()

        plt.savefig(dire + fname + '.pdf', bbox_inches='tight', pad_inches=0, )
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        ax.remove()
        plt.savefig(dire + '/colorbar.pdf', bbox_inches='tight', pad_inches=0, )
        plt.close()
    else:

        #mv = np.percentile(img.flatten(),10)
        #hv = np.percentile(img.flatten(),100)
        tk = 30 / 256
        colors1 = [(256 / 256, 256 / 256, 256 / 256), (tk, tk, tk)]
        map1 = LinearSegmentedColormap.from_list('M1', colors1, N=256)
        plt.register_cmap(cmap=map1)

        colors2 = [(tk, tk, tk), (0/ 256, 0 / 256, 0/ 256)]
        map2 = LinearSegmentedColormap.from_list('M2', colors2, N=256)
        plt.register_cmap(cmap=map2)


        alku= cm.get_cmap('M1', 256)
        alkuvarit = alku(np.linspace(0, 1, 256 * 1))

        loppu= cm.get_cmap('M2', 256)
        loppuvarit = loppu(np.linspace(0, 1, 256 * 1))

        formatter = LogFormatter(10, labelOnlyBase=True,minor_thresholds=(0, 0))

        varivarit = np.vstack((alkuvarit, loppuvarit))

        map4 = ListedColormap(varivarit, name='M4')
        plt.register_cmap(cmap=map4)
        plt.rcParams['image.cmap'] = 'gray_r'
        #maxi = np.max(img)
        #plt.clim(0, maxi)
        plt.imshow(img,norm=mcolors.LogNorm())
        #ax.yaxis.set_major_formatter(formatter)
        #ax.yaxis.set_minor_formatter(formatter)
        # fig.colorbar()
        rc('text', usetex=True)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20,which='both')
        cbar.ax.tick_params(color="red", width=1, length=12)
        cbar.ax.tick_params(which='minor',color="red", width=1, length=5)

        rc('text', usetex=True)


        plt.axis('off')
        # # ax.set_position([0, 0, 1, 1])
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #for tick in ax.get_yticklabels():
        #    tick.set_fontname("Modern Roman")
        plt.tight_layout()
        #rc('text', usetex=True)

        plt.savefig(dire + fname + '.pdf', bbox_inches='tight', pad_inches=0, )

        plt.close()







for fname in os.listdir():
    if fname.endswith('.hdf5'):
        with h5py.File(fname, mode='r') as f:
            column_names = list(f.keys())
            try:
                noise = str(int(f['noise'][()]*100))
                imagename = f['imagefilename'][()]
                imagename = imagename.replace("/","")
                imagename = imagename.replace(".","")
                noisestring = str(round(f['noise'][()],2)).replace('.','')
                # if ('hmc' in method or 'ehmc' in method or 'mwg' in method):
                #     method = 'CM-' + method
                prior  = str(f['prior'][()])
                targetsize = str(f['targetsize'][()])
                tsize = int(targetsize)
                maxangle = str(int(round(f['theta'][()][-1]/45)*45))
                noangles = str(len(f['theta'][()]))
                chain_orig = f['chain'][()]
                cshape = f['chain'].shape
                alpha = f['alpha'][()]
                method2 = f['method'][()]
                #print(method2)
                l1 = f['l1'][()]
                l2 = f['l2'][()]
                duration = f['spent'][()]
            except:
                method2 = 'hmc'
                prior = ''
                targetsize = ''
                tsize = 512
                maxangle = str(180)
                noangles = ''
                noisestring = ''
                fname = fname
                imagename = '120'
                chain_orig = f['chain'][()]
                cshape = f['chain'].shape
            #print(noise,method,prior,targetsize,maxangle,noangles)
            #if (noise=='10'  and method == 'hmc'):
            print(cshape)
            if   ("120" in imagename):
                #data_tuple = (prior, method, targetsize, noise, alpha,maxangle,noangles,l1,l2,duration)
                #cur.execute(lisays, data_tuple)
                # chain = chain_orig[50:,:]
                # column_names = list(f.keys())
                # cor = correlationrow(chain)
                # cor = cor[:,1:30]
                # sums = 1+2*np.sum(cor,axis=1)
                # worst = np.argmax(sums)
                # best = np.argmin(sums)
                # plt.plot(chain_orig[worst,:])
                # plt.plot(chain_orig[best,:])
                # plt.show()
                # exclude target and result

                if  ('hmc' in method2 or 'ehmc' in method2 or 'mwg' in method2):
                    #method = str(f['method'][()])
                    method = 'cm-' + method2
                    #img = f['chain'][()];
                    if (method2 == 'ehmc'):
                        img = np.mean(chain_orig, 1)
                        img = np.reshape(img, (tsize, tsize))
                    elif (method2 == 'hmc'):
                        img = np.mean(chain_orig[:, 100:], 1)
                        img = np.reshape(img, (tsize, tsize))
                    elif (method2 == 'mwg'):
                        img = np.mean(chain_orig[:, 1600:], 1)
                        img = np.reshape(img, (tsize, tsize))
                    else:
                        print("Ei löydy")
                        exit(1)
                    fname = noisestring + '_' + method + '_' + prior + '_' + targetsize + '_' + maxangle + '_' + noangles + '_' + str(imagename)
                    talleta(img,fname)
                    #data_tuple = (prior, method, targetsize, noise, alpha,maxangle,noangles,l1,l2,duration)
                    #cur.execute(lisays, data_tuple)
                    
                    method = 'variance-' + method2
                    
                    if (method2 == 'ehmc'):
                        img = np.var(chain_orig, 1)
                        img = np.reshape(img, (tsize, tsize))
                    elif (method2 == 'hmc'):
                        img = np.var(chain_orig[:, 100:], 1)
                        img = np.reshape(img, (tsize, tsize))
                    elif (method2 == 'mwg'):
                        img = np.var(chain_orig[:, 1600:], 1)
                        img = np.reshape(img, (tsize, tsize))
                    else:
                        print("Ei löydy")
                        exit(1)
                    fname = noisestring + '_' + method + '_' + prior + '_' + targetsize + '_' + maxangle + '_' + noangles  + '_' + str(imagename)
                    #medvar = float(np.median(img))
                    #percentile = float(np.percentile(img,95))
                    #data_tuple = (prior, method, targetsize, noise, alpha, maxangle, noangles, medvar, percentile, duration)
                    #cur.execute(lisays, data_tuple)
                    talleta(img, fname,var=True)




                else:
                    continue
                    method =  method2
                    img = f['result'][()]
                    fname = noisestring + '_' + method + '_' + prior + '_' + targetsize + '_' + maxangle + '_' + noangles  + '_' + str(imagename)
                    talleta(img, fname)
                    data_tuple = (prior, method, targetsize, noise, alpha, maxangle, noangles, l1, l2, duration)
                    #cur.execute(lisays, data_tuple)

            #con.commit()
            elif("mwg" in imagename or "hmc" in imagename):
                print(imagename,fname)
