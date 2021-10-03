import numpy as np
import scipy.linalg as la
import dolfin as dl
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../" )
# sys.path.append('../../../')
from hippylib import *
from reducedHessianSVD import *
from reducedGradient import *
from preconditionedHessianSVD import *
# import matplotlib.pyplot as plt

class GenerateSamples:

    def __init__(self, HiFi, qoi=None, distribution='gaussian', nTotalModes=64):

        self.HiFi = HiFi
        self.qoi = qoi
        self.nTotalModes = nTotalModes

        self.distribution = distribution
        if distribution is 'uniform':
            self.RandomSamples = np.random.uniform(0,1,(1000,self.HiFi.num_para))
        elif distribution is 'gaussian':
            self.RandomSamples = np.random.normal(0,1,(1000,nTotalModes))

        self.d = None # a list of eigenvalues
        self.U = None # a list of eigenvectors
        self.rho = None # a list of scaling factors

        is_fwd_linear = False
        if hasattr(self.HiFi,'problem_type'):
            if self.HiFi.problem_type is not 'nonlinear':
                is_fwd_linear = True

        self.pde = PDEVariationalProblem(self.HiFi.Vh, self.HiFi.residual,
                                         self.HiFi.bc, self.HiFi.bc0, is_fwd_linear = is_fwd_linear)

    def RandomSampling(self, nSamples=1):

        if self.distribution is 'uniform':
            samples = np.zeros([nSamples, self.HiFi.num_para])

            for i in range(self.HiFi.num_para):
                samples[:,i] = np.random.uniform(self.HiFi.para_min[i], self.HiFi.para_max[i], nSamples)

            return samples

        elif self.distribution is 'gaussian':
            samples = []
            noise = dl.Vector()
            self.HiFi.Gauss.init_vector(noise,"noise")
            noise_size = noise.array().shape[0]

            for i in range(nSamples):
                sample = dl.Function(self.HiFi.Vh[1]).vector()
                noise.set_local(np.random.normal(0,1,noise_size))
                self.HiFi.Gauss.sample(noise, sample)
                samples.append(sample.array())
            samples = np.array(samples)

            return samples

    def SubspaceSampling(self, nModes=0, nSamples=1):

        if nModes == 0:
            nModes = self.nTotalModes
        else:
            nModes = np.min([nModes, self.nTotalModes])

        if self.distribution is 'uniform':
            samples = []

            for n in range(nSamples):
                sample = 0.*self.U[:,0]
                para = np.zeros(self.HiFi.num_para)

                for i in range(self.HiFi.num_para):
                    para[i] = self.HiFi.para_min[i] + (self.HiFi.para_max[i] - self.HiFi.para_min[i])*self.RandomSamples[n,i]

                for i in range(nModes):
                    sample += np.dot(para, self.U[:,i])*self.U[:,i]
                samples.append(sample)
            samples = np.array(samples)

            return samples

        elif self.distribution is 'gaussian': # Gaussian random field
            samples = []

            for n in range(nSamples):
                sample = 0.*self.U[:,0]

                for i in range(nModes):
                    sample += self.RandomSamples[n,i]*self.U[:,i]*np.sqrt(self.rho[i])

                samples.append(sample)

            samples = np.array(samples)

            return samples

    def KarhunenLoeve(self, nTotalModes=64):

        if self.distribution is "uniform":
            raise ValueError('uniform distribution is not implemented yet for KL!')

        elif self.distribution is "gaussian":
            randomGen = Random(myid=0)
            m = self.pde.generate_parameter()
            Omega = MultiVector(m, nTotalModes+5)
            for i in xrange(nTotalModes+5):
                randomGen.normal(1., Omega[i])

            # # # preconditioned Hessian
            d, U = doublePass(self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
            Unparray = np.zeros((len(U[0].array()),nTotalModes))
            U = Unparray

        self.d, self.U, self.rho = d, U, d

    def Hessian(self, nTotalModes=64):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        # solve the forward problem at the mean
        u = pde.generate_state()
        m = pde.generate_parameter()
        if self.distribution is 'uniform':
            m.set_local(self.HiFi.para_mean[:])
        elif self.distribution is 'gaussian':
            m.set_local(self.HiFi.Gauss.mean.array())

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

        if self.distribution is 'uniform': # finite dimension uniform random variables
            HessianMat = []
            for i in range(self.HiFi.num_para):
                m = pde.generate_parameter()
                m[i] = 1
                Hessian_m = pde.generate_parameter()
                Hessian.mult(m,Hessian_m)
                HessianMat.append(Hessian_m.array())

            HessianMat = np.array(HessianMat,dtype=float).T
            # d, U = np.linalg.eigh(HessianMat)
            d, U = la.eigh(HessianMat, self.HiFi.CovInv)

            sort_perm = np.abs(d).argsort()
            sort_perm = sort_perm[::-1]
            d = d[sort_perm]
            U = U[:,sort_perm]
            self.d, self.U = d, U

        elif self.distribution is 'gaussian': # infinite-dimensional Gaussian random field
            randomGen = Random(myid=0)
            m = pde.generate_parameter()
            Omega = MultiVector(m, nTotalModes+5)
            for i in xrange(nTotalModes+5):
                randomGen.normal(1., Omega[i])
            # Omega = np.random.randn(noise.array().shape[0], nTotalModes+10)

            # # # preconditioned Hessian
            d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
            Unparray = np.zeros((len(U[0].array()),nTotalModes))
            for i in range(nTotalModes):
                Unparray[:,i]=U[i].array()
            U = Unparray
            self.d, self.U = d, U
            self._ScalingFactor()

    def HessianSet(self, nTotalModes=64, nSamples=10):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        samples = self.RandomSampling(nSamples-1)
        if self.distribution is 'uniform':
            samples = np.append([self.HiFi.para_mean[:]], samples, axis=0)
        elif self.distribution is 'gaussian':
            samples = np.append([self.HiFi.Gauss.mean.array()], samples, axis=0)

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

            if self.distribution is 'uniform': # finite dimension uniform random variables
                HessianMat = []
                for i in range(self.HiFi.num_para):
                    m = pde.generate_parameter()
                    m[i] = 1
                    Hessian_m = pde.generate_parameter()
                    Hessian.mult(m,Hessian_m)
                    HessianMat.append(Hessian_m.array())

                HessianMat = np.array(HessianMat,dtype=float).T
                # d, U = np.linalg.eigh(HessianMat)
                d, U = la.eigh(HessianMat, self.HiFi.CovInv)

                sort_perm = np.abs(d).argsort()
                sort_perm = sort_perm[::-1]
                d = d[sort_perm]
                U = U[:,sort_perm]

            elif self.distribution is 'gaussian': # infinite-dimensional Gaussian random field
                randomGen = Random(myid=0)
                m = pde.generate_parameter()
                Omega = MultiVector(m, nTotalModes+5)
                for i in xrange(nTotalModes+5):
                    randomGen.normal(1., Omega[i])

                # # # preconditioned Hessian
                d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
                Unparray = np.zeros((len(U[0].array()),len(d)))
                for i in range(len(d)):
                    Unparray[:,i]=U[i].array()
                U = Unparray

            for j in range(nTotalModes):
                U_set.append(np.sqrt(np.abs(d[j]))*U[:,j])

        U_set = np.array(U_set).T
        U, d, _ = np.linalg.svd(U_set, full_matrices=False)

        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:,sort_perm]
        self.U, self.d =  U, d

        if self.distribution is 'gaussian':
            self._ScalingFactor()

    def HessianAverage(self, nTotalModes=64, nSamples=10):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        samples = self.RandomSampling(nSamples-1)
        if self.distribution is 'uniform':
            samples = np.append([self.HiFi.para_mean[:]], samples, axis=0)
        elif self.distribution is 'gaussian':
            samples = np.append([self.HiFi.Gauss.mean.array()], samples, axis=0)

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

        if self.distribution is 'uniform': # finite dimension uniform random variables
            HessianMat = []
            for i in range(self.HiFi.num_para):
                m = pde.generate_parameter()
                m[i] = 1
                Hessian_m = pde.generate_parameter()
                Hessian.mult(m,Hessian_m)
                HessianMat.append(Hessian_m.array())

            HessianMat = np.array(HessianMat,dtype=float).T
            # d, U = np.linalg.eigh(HessianMat)
            d, U = la.eigh(HessianMat, self.HiFi.CovInv)

            sort_perm = np.abs(d).argsort()
            sort_perm = sort_perm[::-1]
            d = d[sort_perm]
            U = U[:,sort_perm]
            self.d, self.U = d, U

        elif self.distribution is 'gaussian': # infinite-dimensional Gaussian random field
            randomGen = Random(myid=0)
            m = pde.generate_parameter()
            Omega = MultiVector(m, nTotalModes+5)
            for i in xrange(nTotalModes+5):
                randomGen.normal(1., Omega[i])

            # # # preconditioned Hessian
            d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
            Unparray = np.zeros((len(U[0].array()),len(d)))
            for i in range(len(d)):
                Unparray[:,i]=U[i].array()
            U = Unparray
            self.d, self.U = d, U
            self._ScalingFactor()

    def ActiveSubspace(self, nTotalModes=64, nSamples=0):

        self.nTotalModes = nTotalModes
        if nSamples == 0:
            nSamples = 2*nTotalModes

        qoi = self.qoi
        pde = self.pde

        samples = self.RandomSampling(nSamples-1)
        if self.distribution is 'uniform':
            samples = np.append([self.HiFi.para_mean[:]], samples, axis=0)
        elif self.distribution is 'gaussian':
            samples = np.append([self.HiFi.Gauss.mean.array()], samples, axis=0)

        if self.distribution is 'uniform':

            grad = pde.generate_parameter()
            GradientMat = np.outer(grad.array(),grad.array())
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
                GradientMat += np.outer(grad.array(),grad.array()) / nSamples

            d, U = np.linalg.eigh(GradientMat)
            sort_perm = np.abs(d).argsort()
            sort_perm = sort_perm[::-1]
            d = d[sort_perm]
            U = U[:,sort_perm]

            self.d, self.U = d, U

        elif self.distribution is 'gaussian':

            Gradient = ReducedGradient(pde, qoi, tol=1e-12, samples=samples)

            randomGen = Random(myid=0)
            m = pde.generate_parameter()
            Omega = MultiVector(m, nTotalModes+5)
            for i in xrange(nTotalModes+5):
                randomGen.normal(1., Omega[i])

            # # # preconditioned Gradient
            d, U = doublePassG(Gradient, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
            Unparray = np.zeros((len(U[0].array()),nTotalModes))
            for i in range(nTotalModes):
                Unparray[:,i]=U[i].array()
            U = Unparray
            self.d, self.U = d, U
            self._ScalingFactor()

    def _ScalingFactor(self):

        pde = self.pde
        rho = [] # u^T*M*A{-1}*M*A{-1}*M*u
        for i in range(self.nTotalModes):
            u = pde.generate_parameter()
            self.HiFi.Gauss.init_vector(u,1)
            u.set_local(self.U[:,i])
            x = pde.generate_parameter()
            self.HiFi.Gauss.init_vector(x,1)
            y = pde.generate_parameter()
            self.HiFi.Gauss.init_vector(y,1)
            self.HiFi.Gauss.M.mult(u,y)
            self.HiFi.Gauss.Rsolver.solve(x,y)
            self.HiFi.Gauss.M.mult(x,y)

            rho.append(np.dot(x.array(),y.array()))

        self.rho = rho