import numpy as np
import matplotlib.pyplot as plt

# label_size = 16
# plt.rcParams['xtick.labelsize'] = label_size
# plt.rcParams['ytick.labelsize'] = label_size
# plt.rcParams['axes.labelsize'] = label_size

###############################################################
construction_method = 'POD' # Greedy or POD

filename = 'data/Random' + construction_method + '.npz'
file = np.load(filename)
nRBset          =   file['nRBset']
sigmaPOD        =   file['sigmaPOD']
error_mean      =   file['error_mean']
error_mean_q    =   file['error_mean_q']

labelname = 'Random'
plt.figure(1)
plt.loglog(nRBset,error_mean[nRBset],'k.-', label=labelname)

plt.figure(2)
plt.loglog(nRBset,error_mean_q[nRBset],'k.-', label=labelname)

plt.figure(3)
plt.loglog(sigmaPOD, 'k.-', label=labelname)

######################################################################################
nModesSet = [1, 4, 16, 64]
Subspace = ["Hessian",  "HessianAverage", "HessianSet", "KarhunenLoeve", "ActiveSubspace"]
construction_method = 'POD' # Greedy or POD

TestCase = "HessianVSKarhunenLoeveVSActiveSubspace"
marker = ['bo-','bx-','bd-','b*-']
SubspaceMethod = Subspace[0]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)

marker = ['ro-','rx-','rd-','r*-']
SubspaceMethod = Subspace[3]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)

marker = ['ko-','kx-','kd-','k*-']
SubspaceMethod = Subspace[4]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)


plt.figure(1)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("Error of State")
filename = "figure/ErrorState"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/ErrorState"+TestCase+".eps"
plt.savefig(filename)

plt.close()

plt.figure(2)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("Error of QoI")
filename = "figure/ErrorQoI"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/ErrorQoI"+TestCase+".eps"
plt.savefig(filename)

plt.close()

plt.figure(3)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("POD singular value")
filename = "figure/PODSingularValue"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/PODSingularValue"+TestCase+".eps"
plt.savefig(filename)

plt.close()


############# Hessian
TestCase = "HessianVSHessianSetVSHessianAverage"
marker = ['bo-','bx-','bd-','b*-']
SubspaceMethod = Subspace[0]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)


marker = ['ko-','kx-','kd-','k*-']
SubspaceMethod = Subspace[1]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)


marker = ['ro-','rx-','rd-','r*-']
SubspaceMethod = Subspace[2]
for i in range(len(nModesSet)):
    nModes = nModesSet[i]

    filename = 'data/' + SubspaceMethod + construction_method + 'nModes' + str(nModes) + '.npz'
    file = np.load(filename)
    nRBset          =   file['nRBset']
    sigmaPOD        =   file['sigmaPOD']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod +"_"+str(nModes)
    plt.figure(1)
    plt.loglog(nRBset,error_mean[nRBset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nRBset,error_mean_q[nRBset], marker[i], label=labelname)

    plt.figure(3)
    plt.loglog(sigmaPOD, marker[i], label = labelname)

plt.figure(1)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("Error of State")
filename = "figure/ErrorState"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/ErrorState"+TestCase+".eps"
plt.savefig(filename)
plt.close()

plt.figure(2)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("Error of QoI")
filename = "figure/ErrorQoI"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/ErrorQoI"+TestCase+".eps"
plt.savefig(filename)
plt.close()

plt.figure(3)
plt.legend()
plt.xlabel("# RB basis functions")
plt.ylabel("POD singular value")
filename = "figure/PODSingularValue"+TestCase+".pdf"
plt.savefig(filename)
filename = "figure/PODSingularValue"+TestCase+".eps"
plt.savefig(filename)
plt.close()



######################################################################################
Subspace = ["Hessian", "HessianAverage", "HessianSet", "KarhunenLoeve", "ActiveSubspace"] # "HessianSet",

marker = ['bo-','bx-','bd-','b*-','bs-']

for i in range(5):
    SubspaceMethod = Subspace[i]
    filename = 'data/ParameterReduction' + SubspaceMethod + '.npz'
    file = np.load(filename)

    nPRset = file['nPRset']
    error_mean      =   file['error_mean']
    error_mean_q    =   file['error_mean_q']

    labelname = SubspaceMethod
    plt.figure(1)
    plt.loglog(nPRset, error_mean[nPRset], marker[i], label=labelname)

    plt.figure(2)
    plt.loglog(nPRset, error_mean_q[nPRset], marker[i], label=labelname)

plt.figure(1)
plt.legend()
plt.xlabel("# modes")
plt.ylabel("Error of State")
filename = "figure/ParameterReductionErrorState.pdf"
plt.savefig(filename)
filename = "figure/ParameterReductionErrorState.eps"
plt.savefig(filename)


plt.figure(2)
plt.legend()
plt.xlabel("# modes")
plt.ylabel("Error of QoI")
filename = "figure/ParameterReductionErrorQoI.pdf"
plt.savefig(filename)
filename = "figure/ParameterReductionErrorQoI.eps"
plt.savefig(filename)

plt.close()




# ######################################################################################
#
# nModes = 10
# construction_method = 'POD' # Greedy or POD
# filename = 'figure/Hessian' + construction_method + 'nModes' + str(nModes) + '.npz'
# file = np.load(filename)
# nRBset          =   file['nRBset']
# sigmaPOD        =   file['sigmaPOD']
# error_max       =   file['error_max']
# error_mean      =   file['error_mean']
# error_max_q     =   file['error_max_q']
# error_mean_q    =   file['error_mean_q']
#
# plt.figure(1)
# plt.loglog(nRBset,error_mean[nRBset],'gx-', label='POD, Hessian, 10')
# # plt.loglog(nRBset,error_max[nRBset],'rx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 10')
#
# plt.figure(2)
# plt.loglog(nRBset,error_mean_q[nRBset],'gx-', label='POD, Hessian, 10')
# # plt.loglog(nRBset,error_max_q[nRBset],'rx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 10')
#
# plt.figure(5)
# plt.loglog(sigmaPOD, 'r.-', label = 'Hessian 10')
#
#
#
# ######################################################################################
#
# nModes = 20
# construction_method = 'POD' # Greedy or POD
# filename = 'figure/Hessian' + construction_method + 'nModes' + str(nModes) + '.npz'
# file = np.load(filename)
# nRBset          =   file['nRBset']
# sigmaPOD        =   file['sigmaPOD']
# error_max       =   file['error_max']
# error_mean      =   file['error_mean']
# error_max_q     =   file['error_max_q']
# error_mean_q    =   file['error_mean_q']
#
# plt.figure(1)
# plt.loglog(nRBset,error_mean[nRBset],'ks-', label='POD, Hessian, 20')
# # plt.loglog(nRBset,error_max[nRBset],'kx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 20')
#
# plt.figure(2)
# plt.loglog(nRBset,error_mean_q[nRBset],'ks-', label='POD, Hessian, 20')
# # plt.loglog(nRBset,error_max_q[nRBset],'kx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 20')
#
# plt.figure(5)
# plt.loglog(sigmaPOD, 'k.-', label = 'Hessian 20')
#
#
# plt.figure(1)
# plt.xlabel('# RB basis functions (N)')
# plt.ylabel('relative error')
# plt.legend()
# plt.savefig('figure/POD_randomVShessian_Solution.eps')
# plt.savefig('figure/POD_randomVShessian_Solution.pdf')
# plt.close()
#
# plt.figure(2)
# plt.xlabel('# RB basis functions (N)')
# plt.ylabel('relative error')
# plt.legend()
# plt.savefig('figure/POD_randomVShessian_QoI.eps')
# plt.savefig('figure/POD_randomVShessian_QoI.pdf')
# plt.close()
#
#
# plt.figure(5)
# plt.xlabel('# RB basis functions (N)')
# plt.ylabel('Singular values of SVD snapshots')
# plt.legend()
# plt.savefig('figure/POD_Sigma.eps')
# plt.savefig('figure/POD_Sigma.pdf')
# plt.close()
#
#
#
# plt.figure(8)
# plt.loglog(nRBset,error_mean[nRBset],'ks-', label='POD, Hessian, 20')
# # plt.loglog(nRBset,error_max[nRBset],'gx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 40')
#
# plt.figure(9)
# plt.loglog(nRBset,error_mean_q[nRBset],'ks-', label='POD, Hessian, 20')
# # plt.loglog(nRBset,error_max_q[nRBset],'gx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 40')
#
# # ######################################################################################
# #
# # nModes = 40
# # construction_method = 'POD' # Greedy or POD
# # filename = 'figure/Hessian' + construction_method + 'nModes' + str(nModes) + '.npz'
# # file = np.load(filename)
# # nRBset          =   file['nRBset']
# # sigmaPOD        =   file['sigmaPOD']
# # error_max       =   file['error_max']
# # error_mean      =   file['error_mean']
# # error_max_q     =   file['error_max_q']
# # error_mean_q    =   file['error_mean_q']
# #
# # plt.figure(6)
# # plt.loglog(nRBset,error_mean[nRBset],'g.-', label='$E[||u_{fe}-u_{rb}||]$, POD, Hessian, 40')
# # # plt.loglog(nRBset,error_max[nRBset],'gx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 40')
# #
# # plt.figure(7)
# # plt.loglog(nRBset,error_mean_q[nRBset],'g.-', label='$E[|s_{fe}-s_{rb}|]$, POD, Hessian, 40')
# # # plt.loglog(nRBset,error_max_q[nRBset],'gx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 40')
# #
# # plt.figure(5)
# # plt.loglog(sigmaPOD, 'g.-', label = 'Hessian 40')
# #
# #
# # plt.figure(5)
# # plt.xlabel('# RB basis functions (N)')
# # plt.ylabel('Singular values of SVD snapshots')
# # plt.legend()
# # plt.savefig('figure/POD_Sigma.eps')
# # plt.savefig('figure/POD_Sigma.pdf')
# # plt.close()
# #
# # plt.figure(6)
# # plt.xlabel('# RB basis functions (N)')
# # plt.ylabel('relative error')
# # plt.legend()
# # plt.savefig('figure/POD_HessianModes_Solution.eps')
# # plt.savefig('figure/POD_HessianModes_Solution.pdf')
# # plt.close()
# #
# # plt.figure(7)
# # plt.xlabel('# RB basis functions (N)')
# # plt.ylabel('relative error')
# # plt.legend()
# # plt.savefig('figure/POD_HessianModes_QoI.eps')
# # plt.savefig('figure/POD_HessianModes_QoI.pdf')
# # plt.close()
# #
#
# ######################################################################################
#
# nModes = 20
# construction_method = 'POD' # Greedy or POD
# filename = 'figure/HessianAverage' + construction_method + 'nModes' + str(nModes) + '.npz'
# file = np.load(filename)
# nRBset          =   file['nRBset']
# sigmaPOD        =   file['sigmaPOD']
# error_max       =   file['error_max']
# error_mean      =   file['error_mean']
# error_max_q     =   file['error_max_q']
# error_mean_q    =   file['error_mean_q']
#
# plt.figure(8)
# plt.loglog(nRBset,error_mean[nRBset],'r.-', label='POD, AveragedHessian, 20')
# # plt.loglog(nRBset,error_max[nRBset],'gx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 40')
#
# plt.figure(9)
# plt.loglog(nRBset,error_mean_q[nRBset],'r.-', label='POD, AveragedHessian, 20')
# # plt.loglog(nRBset,error_max_q[nRBset],'gx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 40')
#
#
# ######################################################################################
#
# nModes = 20
# construction_method = 'POD' # Greedy or POD
# filename = 'figure/HessianSet' + construction_method + 'nModes' + str(nModes) + '.npz'
# file = np.load(filename)
# nRBset          =   file['nRBset']
# sigmaPOD        =   file['sigmaPOD']
# error_max       =   file['error_max']
# error_mean      =   file['error_mean']
# error_max_q     =   file['error_max_q']
# error_mean_q    =   file['error_mean_q']
#
# plt.figure(8)
# plt.loglog(nRBset,error_mean[nRBset],'bx-', label='POD, CombinedHessian, 20')
# # plt.loglog(nRBset,error_max[nRBset],'gx-',label='$max[||u_{fe}-u_{rb}||]$, POD, Hessian, 40')
#
# plt.figure(9)
# plt.loglog(nRBset,error_mean_q[nRBset],'bx-', label='POD, CombinedHessian, 20')
# # plt.loglog(nRBset,error_max_q[nRBset],'gx-',label='$max[|s_{fe}-s_{rb}|]$, POD, Hessian, 40')
#
# plt.figure(8)
# plt.xlabel('# RB basis functions (N)')
# plt.ylabel('relative error')
# plt.legend()
# plt.savefig('figure/POD_HessianComparison_Solution.eps')
# plt.savefig('figure/POD_HessianComparison_Solution.pdf')
# plt.close()
#
# plt.figure(9)
# plt.xlabel('# RB basis functions (N)')
# plt.ylabel('relative error')
# plt.legend()
# plt.savefig('figure/POD_HessianComparison_QoI.eps')
# plt.savefig('figure/POD_HessianComparison_QoI.pdf')
# plt.close()
