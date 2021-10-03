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

if plot:
    import matplotlib.pyplot as plt
    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    plt.rcParams['axes.labelsize'] = label_size

space = '='

print space*5, "specify parameters", space*5

nTerms = 256
nTrain = 256
nTest = 20
nMaxRB = 129
nRBset = np.power(2, range(8))
nModesSet = [1,4,16,64]
nTotalModes = 128
nPRset = np.power(2, range(8))

# nTerms = 16
# nTrain = 30
# nTest = 20
# nMaxRB = 12
# nModesSet = [4]
# nPRset = np.power(2,range(3))

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
                                        # Random samples #
print space*5, "generate random training samples", space*5
train_samples = sampling.RandomSampling(nSamples=nTrain)

for construction_method in ['POD']: # 'Greedy'
    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5
    nRBset = np.power(2,range(1+np.int(np.log2(nMaxRB))))

    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/Random' + construction_method + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


######################################################################################################################
                                        # KarhunenLoeve based sample #

print space*5, "create sample generator KarhunenLoeve", space*5
t0 = time.time()
sampling.KarhunenLoeve(nTotalModes = nTotalModes)
print space*5, "KarhunenLoeve creation time = ", time.time() - t0, space*5
np.savez("data/EigenPairsKarhunenLoeve.npz",d=sampling.d,U=sampling.U)

for nModes in nModesSet:
    print space*5, "generate KarhunenLoeve training samples nModes = ", str(nModes), space*5
    train_samples = sampling.SubspaceSampling(nModes=nModes, nSamples=nTrain)

    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    construction_method = 'POD' # Greedy or POD
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5

    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/KarhunenLoeve' + construction_method + 'nModes' + str(nModes) + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/ParameterReductionKarhunenLoeve' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

######################################################################################################################
                                        # Hessian based sample #

print space*5, "create sample generator Hessian", space*5
t0 = time.time()
sampling.Hessian(nTotalModes = nTotalModes)
print space*5, "Hessian creation time = ", time.time() - t0, space*5
np.savez("data/EigenPairsHessian.npz",d=sampling.d,U=sampling.U)

for nModes in nModesSet:
    print space*5, "generate Hessian training samples nModes = ", str(nModes), space*5
    train_samples = sampling.SubspaceSampling(nModes=nModes, nSamples=nTrain)

    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    construction_method = 'POD' # Greedy or POD
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5
    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/Hessian' + construction_method + 'nModes' + str(nModes) + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/ParameterReductionHessian' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


######################################################################################################################
                                        # HessianSet based sample #

print space*5, "create sample generator HessianSet", space*5
t0 = time.time()
sampling.HessianSet(nTotalModes = nTotalModes)
print space*5, "HessianSet creation time = ", time.time() - t0, space*5
np.savez("data/EigenPairsHessianSet.npz",d=sampling.d,U=sampling.U)

for nModes in nModesSet:
    print space*5, "generate HessianSet training samples nModes = ", str(nModes), space*5
    train_samples = sampling.SubspaceSampling(nModes=nModes, nSamples=nTrain)

    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    construction_method = 'POD' # Greedy or POD
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5
    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/HessianSet' + construction_method + 'nModes' + str(nModes) + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)

print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/ParameterReductionHessianSet' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


######################################################################################################################
                                        # HessianAverage based sample #

print space*5, "create sample generator HessianAverage", space*5
t0 = time.time()
sampling.HessianAverage(nTotalModes = nTotalModes)
print space*5, "HessianAverage creation time = ", time.time() - t0, space*5
np.savez("data/EigenPairsHessianAverage.npz",d=sampling.d,U=sampling.U)

for nModes in nModesSet:
    print space*5, "generate HessianAverage training samples nModes = ", str(nModes), space*5
    train_samples = sampling.SubspaceSampling(nModes=nModes, nSamples=nTrain)

    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    construction_method = 'POD' # Greedy or POD
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5
    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/HessianAverage' + construction_method + 'nModes' + str(nModes) + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/ParameterReductionHessianAverage' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


######################################################################################################################
                                        # ActiveSubspace based sample #

print space*5, "create sample generator ActiveSubspace", space*5
t0 = time.time()
sampling.ActiveSubspace(nTotalModes = nTotalModes)
print space*5, "ActiveSubspace creation time = ", time.time() - t0, space*5
np.savez("data/EigenPairsActiveSubspace.npz",d=sampling.d,U=sampling.U)

for nModes in nModesSet:
    print space*5, "generate ActiveSubspace training samples nModes = ", str(nModes), space*5
    train_samples = sampling.SubspaceSampling(nModes=nModes, nSamples=nTrain)

    print space*5, "build reduced basis POD", space*5
    projection_type = 'Galerkin' # Galerkin or PetrovGalerkin
    construction_method = 'POD' # Greedy or POD
    t0 = time.time()
    RB = ReducedBasisSystem(HiFi, train_samples, tol=0, Nmax=nMaxRB,
                            construction_method=construction_method,projection_type=projection_type)
    print space*5, "build reduced basis POD time = ", time.time() - t0, space*5

    print space*5, "test reduced basis error", space*5
    t0 = time.time()
    error_max, error_mean, error_max_q, error_mean_q = errorRB(HiFi, RB, test_samples, nRBset=nRBset, qoi=qoi, plot=plot)
    print space*5, "test time = ", time.time() - t0, space*5

    filename = 'data/ActiveSubspace' + construction_method + 'nModes' + str(nModes) + '.npz'
    np.savez(filename, nRBset=nRBset, sigmaPOD=RB.sigma,
             error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)


print space*5, "test dimension reduction error", space*5
t0 = time.time()
error_max, error_mean, error_max_q, error_mean_q = ErrorParameterReduction(HiFi, test_samples, nSamples=nTest,
                                                                           sampling=sampling, nPRset= nPRset, qoi=qoi, plot=plot)
print space*5, "test time = ", time.time() - t0, space*5

filename = 'data/ParameterReductionActiveSubspace' + '.npz'
np.savez(filename, nPRset=nPRset, error_max=error_max, error_mean=error_mean, error_max_q=error_max_q, error_mean_q=error_mean_q)
