import numpy as np
import scipy as sp
import scipy.linalg as spla

import dolfin as dl
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../" )
# sys.path.append('../../../')
from hippylib import *
from reducedHessianSVD import *
from reducedGradient import *
from preconditionedHessianSVD import *
import matplotlib.pyplot as plt

class GenerateSamplesGaussian:

    def __init__(self, HiFi, qoi=None, nTotalModes=64, nSamples=0, distribution='gaussian'):

        self.HiFi = HiFi
        self.qoi = qoi
        self.nTotalModes = nTotalModes
        self.distribution = distribution

        if nSamples == 0:
            nSamples = 2*nTotalModes

        self.samples = self.RandomSampling(nSamples=nSamples)

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

        samples = []
        noise = dl.Vector()
        self.HiFi.Gauss.init_vector(noise,"noise")
        # noise_size = noise.get_local().shape[0]
        noise_size = noise.local_size()

        for i in range(nSamples):
            sample = dl.Function(self.HiFi.Vh[1]).vector()
            noise.set_local(np.random.normal(0,1,noise_size))
            self.HiFi.Gauss.sample(noise, sample)
            samples.append(sample.get_local())

        samples = np.array(samples)

        return samples

    def SubspaceSampling(self, nModes=0, nSamples=1):

        if nModes == 0:
            nModes = self.nTotalModes
        else:
            nModes = np.min([nModes, self.nTotalModes])

        samples = []

        for n in range(nSamples):
            sample = 0.*self.U[:,0]

            sample_n = self.pde.generate_parameter()
            sample_n.set_local(self.samples[n])
            for i in range(nModes):
                # sample += self.RandomSamples[n,i]*self.U[:,i]*np.sqrt(self.rho[i])
                U_i = self.pde.generate_parameter()
                U_i.set_local(self.U[:,i])
                self.HiFi.Gauss.R.inner(sample_n, U_i)
                sample += self.HiFi.Gauss.R.inner(sample_n, U_i)*self.U[:,i]
            samples.append(sample)

        samples = np.array(samples)

        return samples

    def SubspaceProjection(self, nModes=0, sample=None):
        if nModes == 0:
            nModes = self.nTotalModes
        else:
            nModes = np.min([nModes, self.nTotalModes])

        sampleProjection = 0.*self.U[:,0]

        sample_n = self.pde.generate_parameter()
        sample_n.set_local(sample)
        for i in range(nModes):
            # sample += self.RandomSamples[n,i]*self.U[:,i]*np.sqrt(self.rho[i])
            U_i = self.pde.generate_parameter()
            U_i.set_local(self.U[:,i])
            self.HiFi.Gauss.R.inner(sample_n, U_i)
            sampleProjection += self.HiFi.Gauss.R.inner(sample_n, U_i)*self.U[:,i]

        return sampleProjection

    def KarhunenLoeve(self, nTotalModes=64):

        randomGen = Random(myid=0)
        m = self.pde.generate_parameter()
        Omega = MultiVector(m, nTotalModes+5)
        for i in xrange(nTotalModes+5):
            randomGen.normal(1., Omega[i])

        # # # preconditioned Hessian Rsolver = C
        d, U = doublePassG(self.HiFi.Gauss.Rsolver, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
        Unparray = np.zeros((U[0].local_size(),nTotalModes))
        for i in range(nTotalModes):
            Unparray[:,i]=U[i].get_local()#*np.sqrt(d[i])
        U = Unparray

        # plt.figure()
        # plt.semilogy(np.abs(d),'o')
        # plt.show()

        self.d, self.U = np.sqrt(d), U

    def Hessian(self, nTotalModes=64):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        # solve the forward problem at the mean
        u = pde.generate_state()
        m = pde.generate_parameter()
        m.set_local(self.HiFi.Gauss.mean.get_local())

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

        randomGen = Random(myid=0)
        m = pde.generate_parameter()
        Omega = MultiVector(m, nTotalModes+5)
        for i in xrange(nTotalModes+5):
            randomGen.normal(1., Omega[i])
        # Omega = np.random.randn(noise.get_local().shape[0], nTotalModes+10)

        # # # preconditioned Hessian
        d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
        Unparray = np.zeros((U[0].local_size(),nTotalModes))
        for i in range(nTotalModes):
            Unparray[:,i]=U[i].get_local()
        U = Unparray

        self.d, self.U = d, U

    def HessianSet(self, nTotalModes=64, nSamples=16):

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

            randomGen = Random(myid=0)
            m = pde.generate_parameter()
            Omega = MultiVector(m, nTotalModes+5)
            for j in xrange(nTotalModes+5):
                randomGen.normal(1., Omega[j])

            # # # preconditioned Hessian
            d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
            Unparray = np.zeros((U[0].local_size(),len(d)))
            for j in range(len(d)):
                Unparray[:,j]=U[j].get_local()
            U = Unparray

            for j in range(nTotalModes):
                U_set.append(np.sqrt(np.abs(d[j]))*U[:,j])

        U_set = np.array(U_set).T

        ################ option 1
        ## POD decomposition to compute C^{-1} orthogonal basis

        A = self.HiFi.Gauss.A.array()
        M = self.HiFi.Gauss.M.array()
        C = np.matmul(A, np.linalg.solve(M, A))
        CovMat = np.matmul(U_set.T, np.matmul(C, U_set))

        # N = U_set.shape[1]
        # CovMat = np.zeros((N, N))
        # for n in range(N):
        #     U_n = self.pde.generate_parameter()
        #     U_n.set_local(U_set[:,n])
        #     for m in range(N):
        #         print "m = ", m, "n = ", n
        #         U_m = self.pde.generate_parameter()
        #         U_m.set_local(U_set[:,m])
        #         CovMat[m,n] = self.HiFi.Gauss.R.inner(U_n, U_m)

        d, U = np.linalg.eigh(CovMat)
        sort_perm = np.abs(d).argsort()
        sort_perm = sort_perm[::-1]
        d = d[sort_perm]
        U = U[:,sort_perm]

        # d, U = sp.linalg.eigsh(CovMat, k=self.nTotalModes, which='LM')

        self.U = []
        for i in range(self.nTotalModes):
            self.U.append(1./np.sqrt(np.abs(d[i]))*np.dot(U_set, U[:,i]))

        self.U = np.array(self.U).T

        self.d = d

        ##################### option 2
        # ## SVD + Cholesky decomposition to compute C^{-1} orthogonal basis
        # # C^{-1} = A M^{-1} A = A (L*L^T)^{-1} A = A L^{-T} L^{-1} A
        # # where L*L^T is the Cholesky factorization of the mass matrix
        # import time
        # t0 = time.time()
        # L = np.linalg.cholesky(self.HiFi.Gauss.M.array())
        # print "Cholesky factorization time = ", time.time() - t0
        #
        # t0 = time.time()
        # Linv = np.linalg.inv(L)
        # print "triangular matrix inversion time = ", time.time() - t0
        #
        # A = self.HiFi.Gauss.A.array()
        # t0 = time.time()
        # ALinvU = np.matmul(A, np.matmul(Linv.T, U_set))
        # print "SVD matrix assembling time = ", time.time() - t0
        #
        # t0 = time.time()
        # # S, Sigma, Q = np.linalg.svd(ALinvU, full_matrices=False)
        # ALinvU = sp.sparse.csc_matrix(ALinvU)
        # S, Sigma, Q = sp.sparse.linalg.svds(ALinvU, k=nTotalModes, which='LM')
        # print "SVD computation time = ", time.time() - t0
        #
        # self.U = []
        # for i in range(self.nTotalModes):
        #     # self.U.append(1./Sigma[i]*np.dot(U_set, Q[i,:]))
        #     self.U.append(np.linalg.solve(A, np.dot(L, S[:,i])))
        #
        # self.U = np.array(self.U).T
        # self.d = Sigma[:nTotalModes]**2


        # #################### option 3
        # ## SVD + SquareRoot to compute C^{-1} orthogonal basis
        # # C^{-1} = A M^{-1} A = A (M^{1/2}M^{1/2})^{-1} A
        # import time
        # t0 = time.time()
        # L = spla.sqrtm(self.HiFi.Gauss.M.array())
        # print "matrix square root time = ", time.time() - t0
        #
        # # t0 = time.time()
        # # Linv = np.linalg.inv(L)
        # # print "triangular matrix inversion time = ", time.time() - t0
        #
        # A = self.HiFi.Gauss.A.array()
        # t0 = time.time()
        # ALinvU = np.matmul(A, np.linalg.solve(L, U_set))
        # print "SVD matrix assembling time = ", time.time() - t0
        #
        # t0 = time.time()
        # # S, Sigma, Q = np.linalg.svd(ALinvU, full_matrices=False)
        # ALinvU = sp.sparse.csc_matrix(ALinvU)
        # S, Sigma, Q = sp.sparse.linalg.svds(ALinvU, k=nTotalModes, which='LM')
        # print "SVD computation time = ", time.time() - t0
        #
        # self.U = np.linalg.solve(A, np.dot(L, S))

        # self.U = []
        # for i in range(self.nTotalModes):
        #     # self.U.append(1./Sigma[i]*np.dot(U_set, Q[i,:]))
        #     self.U.append(np.linalg.solve(A, np.dot(L, S[:,i])))
        # self.U = np.array(self.U).T
        # self.d = Sigma[:nTotalModes]


        # # check the orthonamality
        # for i in range(self.nTotalModes):
        #     U_n = self.pde.generate_parameter()
        #     U_n.set_local(self.U[:,i])
        #     print self.HiFi.Gauss.R.inner(U_n, U_n)

        # U = []
        # for i in range(self.nTotalModes):
        #     noise = pde.generate_parameter()
        #     noise.set_local(self.U[i])
        #     sample = pde.generate_parameter()
        #     self.HiFi.Gauss.sample(noise,sample)
        #     U.append(sample.get_local())
        # self.U = np.array(U).T

        ## I orthogonal basis
        # U, d, _ = np.linalg.svd(U_set, full_matrices=False)
        #
        # sort_perm = np.abs(d).argsort()
        # sort_perm = sort_perm[::-1]
        # d = d[sort_perm]
        # U = U[:,sort_perm]
        # self.U, self.d =  U, d

    def HessianAverage(self, nTotalModes=64, nSamples=16):

        self.nTotalModes = nTotalModes
        qoi = self.qoi
        pde = self.pde

        samples = self.samples[:nSamples]

        pde_set = ()
        qoi_set = ()
        for i in range(nSamples):
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

        randomGen = Random(myid=0)
        m = pde.generate_parameter()
        Omega = MultiVector(m, nTotalModes+5)
        for i in xrange(nTotalModes+5):
            randomGen.normal(1., Omega[i])

        # # # preconditioned Hessian
        d, U = doublePassG(Hessian, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
        Unparray = np.zeros((U[0].local_size(),len(d)))
        for i in range(len(d)):
            Unparray[:,i]=U[i].get_local()
        U = Unparray

        self.d, self.U = d, U

    def ActiveSubspace(self, nTotalModes=64, nSamples=0):

        self.nTotalModes = nTotalModes
        if nSamples == 0:
            nSamples = nTotalModes + 10

        qoi = self.qoi
        pde = self.pde

        samples = self.samples[:nSamples]

        # covariance preconditioned active subspace
        Gradient = ReducedGradient(pde, qoi, tol=1e-12, samples=samples)

        randomGen = Random(myid=0)
        m = pde.generate_parameter()
        Omega = MultiVector(m, nTotalModes+5)
        for i in xrange(nTotalModes+5):
            randomGen.normal(1., Omega[i])

        # # # preconditioned Gradient
        d, U = doublePassG(Gradient, self.HiFi.Gauss.R, self.HiFi.Gauss.Rsolver, Omega, nTotalModes)
        Unparray = np.zeros((U[0].local_size(),nTotalModes))
        for i in range(nTotalModes):
            Unparray[:,i]=U[i].get_local()
        U = Unparray

        self.d, self.U = d, U

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

            rho.append(np.dot(x.get_local(),y.get_local()))

        self.rho = rho
