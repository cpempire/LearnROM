import numpy as np
import scipy.linalg as la
import dolfin as dl
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../")
from hippylib import *
from reducedHessianSVD import *
from reducedGradient import *
from preconditionedHessianSVD import *
# import matplotlib.pyplot as plt

class GenerateSamplesUniformV2:

    def __init__(self, HiFi, qoi=None, nTotalModes=64, nSamples=0, distribution='uniform'):

        self.HiFi = HiFi
        self.qoi = qoi
        self.nTotalModes = nTotalModes
        self.distribution = distribution

        if nSamples == 0:
            nSamples = 2*nTotalModes

        self.samples = self.RandomSampling(nSamples=nSamples)

        self.d = None # a list of eigenvalues
        self.U = None # a list of eigenvectors
        self.rho = None # a list of scaling factors

        is_fwd_linear = False
        if hasattr(self.HiFi,'problem_type'):
            if self.HiFi.problem_type is not 'nonlinear':
                is_fwd_linear = True

        self.pde = PDEVariationalProblem(self.HiFi.Vh, self.HiFi.residual,
                                         self.HiFi.bc, self.HiFi.bc0, is_fwd_linear=is_fwd_linear)

    def RandomSampling(self, nSamples=1):

        samples = np.zeros([nSamples, self.HiFi.num_para])

        for i in range(self.HiFi.num_para):
            samples[:,i] = np.random.uniform(self.HiFi.para_min[i], self.HiFi.para_max[i], nSamples)

        return samples

    def SubspaceSampling(self, nModes=0, nSamples=1):

        if nModes == 0:
            nModes = self.nTotalModes
        else:
            nModes = np.min([nModes, self.nTotalModes])

        samples = []

        for n in range(nSamples):
            sample = 0.*self.U[:,0]

            for i in range(nModes):
                sample += np.dot(self.samples[n], self.U[:,i])*self.U[:,i]
            samples.append(sample)

        samples = np.array(samples)

        return samples

    def SubspaceProjection(self, nModes=0, sample=None):
        if nModes == 0:
            nModes = self.nTotalModes
        else:
            nModes = np.min([nModes, self.nTotalModes])

        sampleProjection = 0.*self.U[:,0]

        for i in range(nModes):
            sampleProjection += np.dot(sample, self.U[:,i])*self.U[:,i]

        return sampleProjection

    def KarhunenLoeve(self, nTotalModes=64):

        # d, U = la.eigh(self.HiFi.CovMat, self.HiFi.CovInv) # U^T CovInv U = I
        #
        # sort_perm = np.abs(d).argsort()
        # sort_perm = sort_perm[::-1]
        # d = d[sort_perm]
        # U = U[:,sort_perm]
        #
        # self.d, self.U = d, U

        self.nTotalModes = nTotalModes
        self.U = np.eye(self.HiFi.num_para, nTotalModes)
        self.d = np.ones(self.HiFi.num_para)

    def Hessian(self, nTotalModes=64):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        # solve the forward problem at the mean
        u = pde.generate_state()
        m = pde.generate_parameter()
        m.set_local(self.HiFi.para_mean[:])

        p = pde.generate_state()
        x_all = [u,m,p]
        x = pde.generate_state()
        pde.solveFwd(x,x_all,tol=1e-12)
        x_all[STATE] = x
        # solve the adjoint problem at the mean
        y = pde.generate_state()
        rhs = pde.generate_state()
        qoi.adj_rhs(x_all,rhs)
        pde.solveAdj(y,x_all,rhs,tol=1e-12)
        x_all[ADJOINT] = y

        # set the linearization point for incremental forward and adjoint problems
        pde.setLinearizationPoint(x_all, gauss_newton_approx=False)
        qoi.setLinearizationPoint(x_all)

        Hessian = ReducedHessianSVD(pde, qoi, tol=1e-12)

        HessianMat = []
        for i in range(self.HiFi.num_para):
            m = pde.generate_parameter()
            m[i] = 1
            Hessian_m = pde.generate_parameter()
            Hessian.mult(m,Hessian_m)
            HessianMat.append(Hessian_m.get_local())

        HessianMat = np.array(HessianMat,dtype=float).T
        # d, U = np.linalg.eigh(HessianMat)
        d, U = la.eigh(HessianMat)

        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:,sort_perm]
        self.d, self.U = d, U

    def HessianSet(self, nTotalModes=64, nSamples=32):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        samples = self.samples[:nSamples]

        U_set = []
        for i in range(nSamples):
            # solve the forward problem at the m
            u = pde.generate_state()
            m = pde.generate_parameter()
            m.set_local(samples[i,:])
            p = pde.generate_state()
            x_all = [u,m,p]
            x = pde.generate_state()
            pde.solveFwd(x,x_all,tol=1e-12)
            x_all[STATE] = x
            # solve the adjoint problem at the m
            y = pde.generate_state()
            rhs = pde.generate_state()
            qoi.adj_rhs(x_all,rhs)
            pde.solveAdj(y,x_all,rhs,tol=1e-12)
            x_all[ADJOINT] = y

            # set the linearization point for incremental forward and adjoint problems
            pde.setLinearizationPoint(x_all, gauss_newton_approx=False)
            qoi.setLinearizationPoint(x_all)

            Hessian = ReducedHessianSVD(pde, qoi, tol=1e-12)

            HessianMat = []
            for j in range(self.HiFi.num_para):
                m = pde.generate_parameter()
                m[j] = 1
                Hessian_m = pde.generate_parameter()
                Hessian.mult(m,Hessian_m)
                HessianMat.append(Hessian_m.get_local())

            HessianMat = np.array(HessianMat,dtype=float).T
            # d, U = np.linalg.eigh(HessianMat)
            d, U = la.eigh(HessianMat)

            sort_perm = np.abs(d).argsort()
            sort_perm = sort_perm[::-1]
            d = d[sort_perm]
            U = U[:,sort_perm]

            for j in range(nTotalModes):
                U_set.append(np.sqrt(np.abs(d[j]))*U[:,j])

        U_set = np.array(U_set).T

        # CovMat = np.dot(U_set.T, U_set)
        # d, U = la.eigh(CovMat)
        # sort_perm = np.abs(d).argsort()
        # sort_perm = sort_perm[::-1]
        # d = d[sort_perm]
        # U = U[:,sort_perm]
        #
        # self.U = []
        # for i in range(self.nTotalModes):
        #     self.U.append(1./np.sqrt(np.abs(d[i]))*np.dot(U_set, U[:,i]))
        # self.U = np.array(self.U).T
        #
        # self.d = d

        # I orthogonal basis
        U, d, _ = np.linalg.svd(U_set, full_matrices=False)

        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:, sort_perm]
        self.U, self.d = U, d

    def HessianAverage(self, nTotalModes=64, nSamples=32):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        samples = self.samples[:nSamples]

        pde_set = ()
        qoi_set = ()
        for i in range(len(samples)):
            # solve the adjoint problem at the m
            u = pde.generate_state()
            m = pde.generate_parameter()
            m.set_local(samples[i,:])
            p = pde.generate_state()
            x_all = [u,m,p]
            x = pde.generate_state()
            pde.solveFwd(x,x_all,tol=1e-12)
            x_all[STATE] = x
            # solve the adjoint problem at the m
            y = pde.generate_state()
            rhs = pde.generate_state()
            qoi.adj_rhs(x_all,rhs)
            pde.solveAdj(y,x_all,rhs,tol=1e-12)
            x_all[ADJOINT] = y

            # set the linearization point for incremental forward and adjoint problems
            pde.setLinearizationPoint(x_all, gauss_newton_approx=False)
            qoi.setLinearizationPoint(x_all)

            pde_set += (pde,)
            qoi_set += (qoi,)

        # print "pde_set", pde_set, pde_set[0]

        Hessian = ReducedHessianAverageSVD(pde_set, qoi_set, tol=1e-12, samples=samples)

        HessianMat = []
        for i in range(self.HiFi.num_para):
            m = pde.generate_parameter()
            m[i] = 1
            Hessian_m = pde.generate_parameter()
            Hessian.mult(m, Hessian_m)
            HessianMat.append(Hessian_m.get_local())

        HessianMat = np.array(HessianMat, dtype=float).T
        # d, U = np.linalg.eigh(HessianMat)
        d, U = la.eigh(HessianMat)

        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:,sort_perm]
        self.d, self.U = d, U

    def ActiveSubspace(self, nTotalModes=64, nSamples=0):

        self.nTotalModes = nTotalModes
        if nSamples == 0:
            nSamples = 2*nTotalModes

        qoi = self.qoi
        pde = self.pde

        samples = self.samples[:nSamples]

        # covariance preconditioned active subspace
        grad = pde.generate_parameter()
        GradientMat = np.outer(grad.get_local(), grad.get_local())
        for i in range(nSamples):

            # solve the forward problem at the m
            u = pde.generate_state()
            m = pde.generate_parameter()
            m.set_local(samples[i,:])
            p = pde.generate_state()
            x_all = [u,m,p]
            x = pde.generate_state()
            pde.solveFwd(x,x_all,tol=1e-12)
            x_all[STATE] = x
            # solve the adjoint problem at the m
            y = pde.generate_state()
            rhs = pde.generate_state()
            qoi.adj_rhs(x_all,rhs)
            pde.solveAdj(y,x_all,rhs,tol=1e-12)
            x_all[ADJOINT] = y

            # compute the gradient
            grad = pde.generate_parameter()
            pde.evalGradientParameter(x_all, grad)
            GradientMat += np.outer(grad.get_local(), grad.get_local()) / nSamples

        d, U = la.eigh(GradientMat)
        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:,sort_perm]

        self.d, self.U = d, U