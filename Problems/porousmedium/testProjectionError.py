import numpy as np
import time


from highFidelitySystem import HighFidelitySystem, HighFidelitySystemAdjoint
from qoiPorousMedium import QoIPorousMedium
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../" )
# sys.path.append( "../../../" )
from ReducedBasis import *
from SampleGeneration import *

plot = False

# if plot:
import matplotlib.pyplot as plt

# label_size = 16
# plt.rcParams['xtick.labelsize'] = label_size
# plt.rcParams['ytick.labelsize'] = label_size
# plt.rcParams['axes.labelsize'] = label_size

space = '='

print space*5, "specify parameters", space*5

nTerms = 128
nTest = 10
nTotalModes = 128
nPRset = np.power(2,range(8))

print space*5, "build high fidelity system", space*5
HiFi = HighFidelitySystem(nTerms)

print space*5, "create the QoI", space*5
qoi = QoIPorousMedium(HiFi.mesh,HiFi.Vh[0])

print space*5, "build adjoint high fidelity system", space*5
# HiFiAdjoint = HighFidelitySystemAdjoint(nTerms,qoi)
HiFiAdjoint = None

print space*5, "create sample generator", space*5
sampling = GenerateSamplesUniform(HiFi, qoi=qoi, nTotalModes=nTerms)

print space*5, "generate random test samples", space*5
test_samples = sampling.RandomSampling(nSamples=nTest)

######################################################################################################################
                                        # KarhunenLoeve based sample #

print space*5, "create sample generator KarhunenLoeve", space*5
t0 = time.time()
sampling.KarhunenLoeve(nTotalModes = nTotalModes)
print space*5, "KarhunenLoeve creation time = ", time.time() - t0, space*5
np.savez("data/projection/EigenPairsKarhunenLoeve.npz",d=sampling.d,U=sampling.U)

# testOrth = []
# UiCinvUi = np.zeros(sampling.nTotalModes)
# for i in range(sampling.nTotalModes):
#     U_i = sampling.U[:,i]
#     UiCinvUi[i] = np.dot(np.dot(HiFi.CovInv, U_i), U_i)
#
# testOrth.append(UiCinvUi)
#
# testL2Error = []
# L2Error = np.zeros((sampling.nTotalModes, nTest))
# for j in range(nTest):
#     sample = test_samples[j]
#     for i in nPRset:
#         sampleProjection = sampling.SubspaceProjection(nModes=i, sample=sample)
#         sampleDiff = sample-sampleProjection
#         L2Error[i-1,j] = np.dot(sampleDiff, sampleDiff)
# testL2Error.append(L2Error)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/projection/ParameterReductionKarhunenLoeve' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

######################################################################################################################
                                        # Hessian based sample #

print space*5, "create sample generator Hessian", space*5
t0 = time.time()
sampling.Hessian(nTotalModes = nTotalModes)
print space*5, "Hessian creation time = ", time.time() - t0, space*5
np.savez("data/projection/EigenPairsHessian.npz",d=sampling.d,U=sampling.U)

# UiCinvUi = np.zeros(sampling.nTotalModes)
# for i in range(sampling.nTotalModes):
#     U_i = sampling.U[:,i]
#     UiCinvUi[i] = np.dot(np.dot(HiFi.CovInv, U_i), U_i)
# testOrth.append(UiCinvUi)
#
# L2Error = np.zeros((sampling.nTotalModes, nTest))
# for j in range(nTest):
#     sample = test_samples[j]
#     for i in nPRset:
#         sampleProjection = sampling.SubspaceProjection(nModes=i, sample=sample)
#         sampleDiff = sample-sampleProjection
#         L2Error[i-1,j] = np.dot(sampleDiff, sampleDiff)
# testL2Error.append(L2Error)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/projection/ParameterReductionHessian' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

######################################################################################################################
                                        # HessianSet based sample #

print space*5, "create sample generator HessianSet", space*5
t0 = time.time()
sampling.HessianSet(nTotalModes = nTotalModes)
print space*5, "HessianSet creation time = ", time.time() - t0, space*5
np.savez("data/projection/EigenPairsHessianSet.npz",d=sampling.d,U=sampling.U)

# UiCinvUi = np.zeros(sampling.nTotalModes)
# for i in range(sampling.nTotalModes):
#     U_i = sampling.U[:,i]
#     UiCinvUi[i] = np.dot(np.dot(HiFi.CovInv, U_i), U_i)
# testOrth.append(UiCinvUi)
#
# L2Error = np.zeros((sampling.nTotalModes, nTest))
# for j in range(nTest):
#     sample = test_samples[j]
#     for i in nPRset:
#         sampleProjection = sampling.SubspaceProjection(nModes=i, sample=sample)
#         sampleDiff = sample-sampleProjection
#         L2Error[i-1,j] = np.dot(sampleDiff, sampleDiff)
# testL2Error.append(L2Error)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/projection/ParameterReductionHessianSet' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

#####################################################################################################################
                                        # HessianAverage based sample #

print space*5, "create sample generator HessianAverage", space*5
t0 = time.time()
sampling.HessianAverage(nTotalModes = nTotalModes)
print space*5, "HessianAverage creation time = ", time.time() - t0, space*5
np.savez("data/projection/EigenPairsHessianAverage.npz",d=sampling.d,U=sampling.U)

# UiCinvUi = np.zeros(sampling.nTotalModes)
# for i in range(sampling.nTotalModes):
#     U_i = sampling.U[:,i]
#     UiCinvUi[i] = np.dot(np.dot(HiFi.CovInv, U_i), U_i)
# testOrth.append(UiCinvUi)
#
# L2Error = np.zeros((sampling.nTotalModes, nTest))
# for j in range(nTest):
#     sample = test_samples[j]
#     for i in nPRset:
#         sampleProjection = sampling.SubspaceProjection(nModes=i, sample=sample)
#         sampleDiff = sample-sampleProjection
#         L2Error[i-1,j] = np.dot(sampleDiff, sampleDiff)
# testL2Error.append(L2Error)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/projection/ParameterReductionHessianAverage' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

######################################################################################################################
                                        # ActiveSubspace based sample #

print space*5, "create sample generator ActiveSubspace", space*5
t0 = time.time()
sampling.ActiveSubspace(nTotalModes = nTotalModes)
print space*5, "ActiveSubspace creation time = ", time.time() - t0, space*5
np.savez("data/projection/EigenPairsActiveSubspace.npz",d=sampling.d,U=sampling.U)

# UiCinvUi = np.zeros(sampling.nTotalModes)
# for i in range(sampling.nTotalModes):
#     U_i = sampling.U[:,i]
#     UiCinvUi[i] = np.dot(np.dot(HiFi.CovInv, U_i), U_i)
# testOrth.append(UiCinvUi)
#
# L2Error = np.zeros((sampling.nTotalModes, nTest))
# for j in range(nTest):
#     sample = test_samples[j]
#     for i in nPRset:
#         sampleProjection = sampling.SubspaceProjection(nModes=i, sample=sample)
#         sampleDiff = sample-sampleProjection
#         L2Error[i-1,j] = np.dot(sampleDiff, sampleDiff)
# testL2Error.append(L2Error)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/projection/ParameterReductionActiveSubspace' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

# ######################################################################################################################
#
# Subspace = ["KarhunenLoeve", "Hessian", "HessianAverage", "ActiveSubspace"] # "HessianSet",
# marker = ['bo-','bx-','bd-','b*-','ro-','rx-','rd-','r*-']
#
# # plt.figure()
# # for i in range(4):
# #     plt.plot(testOrth[i],marker[i],label=Subspace[i])
# # plt.xlabel("# modes")
# # plt.legend()
#
# for i in range(4):
#     L2Error = testL2Error[i]
#     L2ErrorMax = np.max(L2Error,axis=1)
#     L2ErrorMean = np.mean(L2Error,axis=1)
#     plt.figure(1)
#     plt.loglog(nPRset, L2ErrorMean[nPRset-1], marker[i], label=Subspace[i])
#     plt.figure(2)
#     plt.loglog(nPRset, L2ErrorMax[nPRset-1], marker[i+4], label=Subspace[i])
#
# plt.figure(1)
# plt.xlabel("# modes")
# plt.ylabel("Mean of L2 error ")
# plt.legend()
# filename = "figure/projection/MeanL2Error.pdf"
# plt.savefig(filename)
# filename = "figure/projection/MeanL2Error.eps"
# plt.savefig(filename)
#
# plt.figure(2)
# plt.xlabel("# modes")
# plt.ylabel("Mean of L2 error ")
# plt.legend()
# filename = "figure/projection/MaxL2Error.pdf"
# plt.savefig(filename)
# filename = "figure/projection/MaxL2Error.eps"
# plt.savefig(filename)
#
# plt.close()