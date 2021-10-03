import numpy as np
import dolfin as dl
from basisConstructionPOD import BasisConstructionPOD
from systemProjection import SystemProjection
import matplotlib.pyplot as plt
import time

class BuildReducedBasisPOD:

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
        print "generate snapshots"
        t = time.clock()
        snapshots = HiFi.snapshots(samples)
        print "generate snapshots time = ", time.clock()-t

        # compute POD basis functions
        snapshots_inner = snapshots[self.node_inner, :]
        basisPOD = BasisConstructionPOD(snapshots_inner,'SVD',tol=tol,Nmax=Nmax)
        self.basis = basisPOD.basis
        self.sigma = basisPOD.sigma
        self.N = basisPOD.N

        # project the system
        self.method = method
        systemProjection = SystemProjection(HiFi, self.basis, self.method)
        self.A_rb = systemProjection.A_rb
        self.f_rb = systemProjection.f_rb

    def solve(self, sample, N=0, assemble=True):
        coeff_a, coeff_f = self.coeffEvaluation(sample)
        if hasattr(self, 'problem_type'):
            if self.problem_type == 'nonaffine' and assemble:
                # print "assemble", assemble
                # assemble HiFi matrix and vector and project them to RB matrix and vector
                self.HiFi.assemble(sample)
                systemProjection = SystemProjection(self.HiFi, self.basis, self.method)
                self.A_rb = systemProjection.A_rb
                self.f_rb = systemProjection.f_rb

        if N == 0:
            N = self.N

        if self.method == 'Galerkin':
            A = coeff_a[0]*self.A_rb[0][:N, :N]
            for i in range(1, self.Qa):
                A += coeff_a[i]*self.A_rb[i][:N, :N]

            f = coeff_f[0]*self.f_rb[0][:N]
            for i in range(1, self.Qf):
                f += coeff_f[i]*self.f_rb[i][:N]

        elif self.method == 'PetrovGalerkin':

            A = 0.*self.A_rb[0][:N, :N]
            f = 0.*self.f_rb[0][:N]
            for i in range(self.Qa):
                for j in range(self.Qa):
                    A += coeff_a[i]*coeff_a[j]*self.A_rb[i*self.Qa+j][:N,:N]
                for j in range(self.Qf):
                    f += coeff_a[i]*coeff_f[j]*self.f_rb[i*self.Qa+j][:N]
        else:
            raise NameError('choose between Galerkin and PetrovGalerkin')

        uN = np.linalg.solve(A, f)

        return uN

    def reconstruct(self, sample, N, assemble=True):
        uN = self.solve(sample, N, assemble)

        u_rb = uN[0]*self.basis[:,0]
        for i in range(1, N):
            u_rb += uN[i]*self.basis[:,i]

        # print "dofs",self.dofs
        solution = np.zeros(self.dofs)
        solution[self.node_inner] = u_rb
        solution[self.node_Dirichlet] = self.u_Dirichlet


        u_fun = dl.Function(self.HiFi.Vh[0])
        u_fun.vector().set_local(solution)
        dl.plot(u_fun)
        dl.interactive()

        return solution

    def snapshots(self, samples, N=0, assemble=True):
        if N == 0:
            N = self.N

        num_samples = samples.shape[0]

        U = []
        for i in range(num_samples):
            sample = samples[i, :]
            u = self.reconstruct(sample, N, assemble)
            U.append(u)
        U = np.array(U).T
        return U