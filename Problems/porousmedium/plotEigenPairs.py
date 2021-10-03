from pylab import *
import numpy as np
import matplotlib.pyplot as plt
label_size = 16
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size
plt.rcParams['axes.labelsize'] = label_size

filename = ["Hessian", "HessianAverage", "HessianSet", "KarhunenLoeve","ActiveSubspace"]

for i in range(5):
    file_i = "data/EigenPairs" + filename[i]+ ".npz"
    data = np.load(file_i)
    d = data['d']
    U = data['U']

    plt.figure()
    indexplus = np.where(d > 0)[0]
    plt.semilogy(indexplus, d[indexplus], 'ro',label='positive')
    indexminus = np.where(d < 0)[0]
    plt.semilogy(indexminus, -d[indexminus], 'kx',label='negative')
    plt.legend()
    plt.xlabel('# eigenvalue')
    plt.ylabel('eigenvalue')
    file_i = "figure/EigenValue" + filename[i]+ ".pdf"
    plt.savefig(file_i)
    file_i = "figure/EigenValue" + filename[i]+ ".eps"
    plt.savefig(file_i)

    m,n = U.shape
    for j in range(0,np.min([64,n]),3):
        xlen = np.sqrt(m)
        x = np.linspace(0,1,xlen)
        X,Y = np.meshgrid(x,x)
        Z = U[:,j].reshape((xlen,xlen))

        fig, ax = plt.subplots()

        p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=-1., vmax=1.)
        cb = fig.colorbar(p)

        file_ij = "figure/EigenVector" + filename[i] + "_Mode_" + str(j) + ".pdf"
        fig.savefig(file_ij)
        file_ij = "figure/EigenVector" + filename[i] + "_Mode_" + str(j) + ".eps"
        fig.savefig(file_ij)
        plt.close()
