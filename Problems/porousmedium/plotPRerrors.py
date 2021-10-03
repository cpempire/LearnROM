import numpy as np
import matplotlib.pyplot as plt

# label_size = 16
# plt.rcParams['xtick.labelsize'] = label_size
# plt.rcParams['ytick.labelsize'] = label_size
# plt.rcParams['axes.labelsize'] = label_size


######################################################################################
Subspace = ["KarhunenLoeve", "Hessian", "HessianSet", "HessianAverage", "ActiveSubspace"] # "HessianSet",

nPRset = np.power(2,range(8))

marker = ['bo-','bx-','bd-','b*-','bs-']

for i in range(5):
    SubspaceMethod = Subspace[i]
    filename = 'data/projection/ParameterReduction' + SubspaceMethod + '.npz'
    file = np.load(filename)

    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod
    plt.figure(1)
    plt.loglog(nPRset,error_mean[nPRset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nPRset,error_mean_q[nPRset], marker[i], label=labelname)

plt.figure(1)
plt.legend()
plt.xlabel("# modes")
plt.ylabel("Error of State")
filename = "figure/projection/ParameterReductionErrorState.pdf"
plt.savefig(filename)
filename = "figure/projection/ParameterReductionErrorState.eps"
plt.savefig(filename)


plt.figure(2)
plt.legend()
plt.xlabel("# modes")
plt.ylabel("Error of QoI")
filename = "figure/projection/ParameterReductionErrorQoI.pdf"
plt.savefig(filename)
filename = "figure/projection/ParameterReductionErrorQoI.eps"
plt.savefig(filename)

plt.close()
