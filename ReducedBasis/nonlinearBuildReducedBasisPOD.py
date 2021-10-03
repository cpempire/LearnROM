import numpy as np
import dolfin as dl
from .basisConstructionPOD import BasisConstructionPOD
from .nonlinearReducedBasisSolver import NonLinearReducedBasisSolver
from .systemProjection import SystemProjection
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../" )
from HyperReduction import *

class NonLinearBuildReducedBasisPOD:

    def __init__(self, HiFi, samples, method, tol, Nmax):

        # parameters
        # self.num_para  = HiFi.num_para
        # self.para_mean = HiFi.para_mean
        # self.para_min  = HiFi.para_min
        # self.para_max  = HiFi.para_max
        if hasattr(HiFi, 'problem_type'):
            self.problem_type = HiFi.problem_type
            self.HiFi = HiFi

        self.Qa        = HiFi.Qa
        self.Qf        = HiFi.Qf

        # boundary/inner index and data
        self.node_inner = HiFi.node_inner
        self.node_Dirichlet = HiFi.node_Dirichlet
        self.dofs = HiFi.dofs
        self.u_Dirichlet = HiFi.u_Dirichlet

        # coefficient evaluation
        self.coeffEvaluation = HiFi.coeffEvaluation

        # compute snapshots at training samples
        U, dU, R = HiFi.snapshots(samples, output='all')

        # compute POD basis functions
        dU_inner = dU[self.node_inner, :]
        self.basis, self.sigma, self.N = BasisConstructionPOD(dU_inner,'SVD',tol=tol,Nmax=Nmax)
        # self.basis = basisPOD.basis
        # self.sigma = basisPOD.sigma
        # self.N = basisPOD.N

        self.solver = NonLinearReducedBasisSolver(self.HiFi.problem, self.basis)

        # import matplotlib.pyplot as plt
        # R_inner = R[self.node_inner, :]
        # basisEIM = BasisConstructionPOD(R_inner,'SVD',tol=tol,Nmax=Nmax)
        # plt.figure()
        # plt.semilogy(basisPOD.sigma,'r.-',label='POD sigma')
        # plt.semilogy(basisEIM.sigma,'k*-',label='EiM sigma')
        # plt.legend()
        # plt.show()


        # construct hyper reduction
        print "construct hyper reduction"
        # U, dU, R = self.snapshots(samples, self.N)
        R = R[HiFi.node_inner,:]
        self.solver.DEIM = DiscreteEmpiricalInterpolation(R,tol=1e-26)

        print "DEIM basis", self.solver.DEIM.N

    # def reconstruct(self, sample, N, assemble=True):
    #     uN = self.solver.solve(sample, N, assemble)
    #
    #     u_rb = uN[0]*self.basis[:,0]
    #     for i in range(1, N):
    #         u_rb += uN[i]*self.basis[:,i]
    #
    #     # print "dofs",self.dofs
    #     solution = np.zeros(self.dofs)
    #     solution[self.node_inner] = u_rb
    #     solution[self.node_Dirichlet] = self.u_Dirichlet
    #
    #     return solution

    def snapshots(self, samples, N, output='all'):

        num_samples = samples.shape[0]

        U = ()
        dU = ()
        R = ()
        m = dl.Function(self.HiFi.Vh[1])
        u = dl.Function(self.HiFi.Vh[0])

        for i in range(num_samples):
            sample = samples[i, :]
            m.vector().set_local(sample)
            self.solver.extra_args = (m,)

            u0 = np.zeros(N)
            solution, dsolution, residual = self.solver.solve(self.HiFi, u0, N)

            dU += dsolution
            R  += residual

            # u.vector().set_local(solution)
            # dl.plot(u)

            U += (solution,)

        # dl.interactive()

        U = np.array(U).T
        dU = np.array(dU).T
        R = np.array(R).T

        if output == 'solution':
            return U
        else:
            return U, dU, R

    def snapshotsDEIM(self, samples, N, M, output='all'):

        num_samples = samples.shape[0]

        U = ()
        dU = ()
        R = ()
        m = dl.Function(self.HiFi.Vh[1])
        u = dl.Function(self.HiFi.Vh[0])

        for i in range(num_samples):
            sample = samples[i, :]
            m.vector().set_local(sample)
            self.solver.extra_args = (m,)

            u0 = np.zeros(N)
            solution, dsolution, residual = self.solver.solve(self.HiFi, u0, N, M)

            dU += dsolution
            R  += residual

            # u.vector().set_local(solution)
            # dl.plot(u)

            U += (solution,)

        # dl.interactive()

        U = np.array(U).T
        dU = np.array(dU).T
        R = np.array(R).T

        if output == 'solution':
            return U
        else:
            return U, dU, R
