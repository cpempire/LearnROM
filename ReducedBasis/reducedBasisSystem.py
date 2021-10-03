import numpy as np
from .buildReducedBasisPOD import BuildReducedBasisPOD
from .buildReducedBasisGreedy import BuildReducedBasisGreedy, UpdateReducedBasisGreedy
from .systemProjection import SystemProjection


class ReducedBasisSystem:

    def __init__(self, HiFi, samples, tol, Nmax, construction_method='POD', projection_type='Galerkin'):

        self.HiFi = HiFi
        self.tol = tol
        self.tol_init = tol
        self.Nmax = Nmax
        self.construction_method = construction_method
        self.projection_type = projection_type

        # parameters
        if hasattr(HiFi, 'problem_type'):
            self.problem_type = HiFi.problem_type

        # boundary/inner index and data
        self.node_inner = HiFi.node_inner
        self.node_Dirichlet = HiFi.node_Dirichlet
        self.dofs = HiFi.dofs
        self.u_Dirichlet = HiFi.u_Dirichlet

        self.Qa_rb = HiFi.Qa_rb
        self.Qf_rb = HiFi.Qf_rb

        # build the reduced basis system
        self.N = None
        self.N_history = []
        self.basis = None
        self.A_rb = None
        self.f_rb = None
        self.sigma = None

        if HiFi.misfit is not None:
            self.basisAdj = None
            self.At_rb = None

        self.build(samples)

    def build(self, samples):

        if self.construction_method is 'POD':
            BuildReducedBasisPOD(self, self.HiFi, samples, self.tol, self.Nmax)
            # self.basis, self.sigma, self.N = BuildReducedBasisPOD(HiFi, samples, tol, Nmax)
            # self.A_rb, self.f_rb = SystemProjection(HiFi, self.basis, projection_type)

        elif self.construction_method is 'Greedy':
            BuildReducedBasisGreedy(self, samples, self.tol, self.Nmax)
            # if (self.HiFiAdjoint is None) and (self.HiFiMisfit is None):
            #     BuildReducedBasisGreedy(self, self.HiFi, samples, self.tol, self.Nmax)
            # elif self.HiFiAdjoint is not None: # goal-oriented greedy construction
            #     BuildReducedBasisGreedyQoI(self, self.HiFi, self.HiFiAdjoint, samples, self.tol, self.Nmax)
            # elif self.HiFiMisfit is not None:
            #     BuildReducedBasisGreedyMisfit(self, self.HiFi, self.HiFiMisfit, samples, self.tol, self.Nmax)
            # self.basis, self.sigma, self.N = BuildReducedBasisGreedy(self, HiFi, samples, tol, Nmax)
        else:
            raise NameError("construction_method is 'POD' or 'Greedy' ")

    def update(self, samples, restart=False, tol=1.):

        self.tol = tol * self.tol_init

        # print("self.tol", self.tol, "self.tol_init", self.tol_init, "tol", tol)

        if self.construction_method is 'POD':
            BuildReducedBasisPOD(self, self.HiFi, samples, self.tol, self.Nmax)

        elif self.construction_method is 'Greedy':
            UpdateReducedBasisGreedy(self, samples, restart)
            # if (self.HiFiAdjoint is None) and (self.HiFiMisfit is None):
            #     UpdateReducedBasisGreedy(self, samples, restart)
            # elif self.HiFiAdjoint is not None: # goal-oriented greedy construction
            #     UpdateReducedBasisGreedyQoI(self, samples, restart)
            # elif self.HiFiMisfit is not None:
            #     UpdateReducedBasisGreedyMisfit(self, samples, restart)
            # # self.basis, self.sigma, self.N = BuildReducedBasisGreedy(self, HiFi, samples, tol, Nmax)
        else:
            raise NameError("construction_method is 'POD' or 'Greedy' ")

        self.N_history.append(self.N)

    def coeffEvaluation(self, sample):
        """
        Evaluate the coefficients of the affine terms
        :param sample: np.random.uniform(-1,1,num_para)
        :return: coefficients
        """

        coeff_a, coeff_f = self.HiFi.coeffEvaluation(sample)

        if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 1e-15:
            for i in range(self.Qa_rb):
                coeff_f.append(-coeff_a[i])

        return coeff_a, coeff_f

    def coeffGradient(self, sample):
        """
        Evaluate the gradient of coefficients of the affine terms
        :param sample: parameter sample
        :return: gradient of the coefficients
        """
        coeff_grad_a, coeff_grad_f = self.HiFi.coeffGradient(sample)
        if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 1e-15:
            for i in range(self.Qa_rb):
                coeff_grad_f.append(-coeff_grad_a[i])

        return coeff_grad_a, coeff_grad_f

    def solve(self, sample, N=0, assemble=True):

        coeff_a, coeff_f = self.coeffEvaluation(sample)

        # if hasattr(self, 'problem_type'):
        #     if self.problem_type is 'nonaffine' and assemble:
        #         # assemble HiFi matrix and vector and project them to RB matrix and vector
        #         self.HiFi.assemble(sample)
        #         self.A_rb, self.f_rb = SystemProjection(self, self.basis)

        if N == 0:
            N = self.N

        if self.projection_type == 'Galerkin':
            A = coeff_a[0]*self.A_rb[0][:N, :N]
            for i in range(1, self.Qa_rb):
                A += coeff_a[i]*self.A_rb[i][:N, :N]

            f = coeff_f[0]*self.f_rb[0][:N]
            for i in range(1, self.Qf_rb):
                f += coeff_f[i]*self.f_rb[i][:N]

            # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
            #     for i in range(self.Qa):
            #         f -= coeff_a[i]*self.f_rb[self.Qf+i][:N]

        elif self.projection_type == 'PetrovGalerkin':

            A = 0.*self.A_rb[0][:N, :N]
            f = 0.*self.f_rb[0][:N]
            for i in range(self.Qa_rb):
                for j in range(self.Qa_rb):
                    A += coeff_a[i]*coeff_a[j]*self.A_rb[i*self.Qa_rb+j][:N,:N]
                for j in range(self.Qf_rb):
                    f += coeff_a[i]*coeff_f[j]*self.f_rb[i*self.Qa_rb+j][:N]

                # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
                #     for j in range(self.Qa):
                #         f -= coeff_a[j]*self.f_rb[self.Qf+j][:N]

        else:
            raise NameError('choose between Galerkin and PetrovGalerkin')

        uN = np.linalg.solve(A, f)

        # self.uN = uN

        return uN

    def reconstruct(self, sample, N=0, assemble=True):
        if N == 0:
            N = self.N

        uN = self.solve(sample, N, assemble)

        u_rb = uN[0]*self.basis[:,0]
        for i in range(1, N):
            u_rb += uN[i]*self.basis[:,i]

        # # print "dofs",self.dofs
        solution = np.zeros(self.dofs)
        solution[self.node_inner] = u_rb
        solution[self.node_Dirichlet] = self.u_Dirichlet

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

    def error_estimate(self, sample, N=0, assemble=True):

        if N == 0:
            N = self.N

        uN = self.solve(sample, N, assemble)
        coeff_a, coeff_f = self.coeffEvaluation(sample)

        # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 1e-15:
        #     for i in range(self.Qa_rb):
        #         coeff_f.append(-coeff_a[i])

        res_aa = 0.
        res_af = 0.
        res_ff = 0.

        # print "self.Qf_rb, len(coeff_f), len(self.fj_Xinv_fi)", self.Qf_rb, len(coeff_f), len(self.fj_Xinv_fi)
        for i in range(self.Qf_rb):
            for j in range(self.Qf_rb):
                res_ff += coeff_f[i]*coeff_f[j]*self.fj_Xinv_fi[self.Qf_rb*i+j]

        for i in range(self.Qa_rb):
            for j in range(self.Qf_rb):
                res_af += coeff_f[j]*coeff_a[i]*np.dot(self.fj_Xinv_Ai_V[self.Qf_rb*i+j][:N], uN)

            for j in range(self.Qa_rb):
                res_aa += coeff_a[j]*coeff_a[i]*np.dot(uN, np.dot(self.Vt_Aj_Xinv_Ai_V[self.Qa_rb*i+j][:N, :N], uN))

        # print "res_aa, res_af, res_ff", res_aa, res_af, res_ff

        res = np.sqrt(np.abs(res_aa - 2*res_af + res_ff))

        # # test accuracy
        # u_rb = self.reconstruct(sample, self.N)[self.HiFi.node_inner]
        # u = self.HiFi.solve(sample)[self.HiFi.node_inner]
        # print("u-u_rb", np.sqrt(np.dot(self.HiFi.Xnorm_inner.dot(u-u_rb).T, u-u_rb))/np.sqrt(np.dot(self.HiFi.Xnorm_inner.dot(u).T, u)), "res", res)

        return res



    # ################################ for adjoint problem #################################################################
    #
    # def coeffEvaluationAdjoint(self,sample):
    #     """
    #     Evaluate the coefficients of the affine terms
    #     :param sample: np.random.uniform(-1,1,num_para)
    #     :return: coefficients
    #     """
    #
    #     coeff_a, _ = self.HiFi.coeffEvaluation(sample)
    #
    #     coeff_f = [1.]
    #
    #     return coeff_a, coeff_f
    #
    # def solveAdjoint(self, sample, N=0, assemble=True):
    #     coeff_a, coeff_f = self.coeffEvaluationAdjoint(sample)
    #     if hasattr(self, 'problem_type'):
    #         if self.problem_type == 'nonaffine' and assemble:
    #             # print "assemble", assemble
    #             # assemble HiFi matrix and vector and project them to RB matrix and vector
    #             self.HiFiAdjoint.assemble(sample)
    #             self.A_rb_adjoint, self.f_rb_adjoint = SystemProjection(self.HiFiAdjoint, self.basis_adjoint, self.projection_type)
    #
    #     if N == 0:
    #         N = self.N
    #
    #     if self.projection_type == 'Galerkin':
    #         A = coeff_a[0]*self.A_rb_adjoint[0][:N, :N]
    #         for i in range(1, self.HiFiAdjoint.Qa_rb):
    #             A += coeff_a[i]*self.A_rb_adjoint[i][:N, :N]
    #
    #         f = coeff_f[0]*self.f_rb_adjoint[0][:N]
    #         for i in range(1, self.HiFiAdjoint.Qf_rb):
    #             f += coeff_f[i]*self.f_rb_adjoint[i][:N]
    #
    #         # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
    #         #     for i in range(self.Qa):
    #         #         f -= coeff_a[i]*self.f_rb[self.Qf+i][:N]
    #
    #     elif self.projection_type == 'PetrovGalerkin':
    #
    #         A = 0.*self.A_rb_adjoint[0][:N, :N]
    #         f = 0.*self.f_rb_adjoint[0][:N]
    #         for i in range(self.HiFiAdjoint.Qa_rb):
    #             for j in range(self.HiFiAdjoint.Qa_rb):
    #                 A += coeff_a[i]*coeff_a[j]*self.A_rb_adjoint[i*self.HiFiAdjoint.Qa_rb+j][:N,:N]
    #             for j in range(self.HiFiAdjoint.Qf_rb):
    #                 f += coeff_a[i]*coeff_f[j]*self.f_rb_adjoint[i*self.HiFiAdjoint.Qa_rb+j][:N]
    #
    #             # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
    #             #     for j in range(self.Qa):
    #             #         f -= coeff_a[j]*self.f_rb[self.Qf+j][:N]
    #
    #     else:
    #         raise NameError('choose between Galerkin and PetrovGalerkin')
    #
    #     pN = np.linalg.solve(A, f)
    #
    #     return pN
    #
    # def error_estimate_qoi(self, sample, N=0, assemble=True):
    #
    #     if N is 0:
    #         N = self.N
    #
    #     coeff_a, coeff_f = self.coeffEvaluation(sample)
    #     # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
    #     #     for i in range(self.Qa_rb):
    #     #         coeff_f.append(-coeff_a[i])
    #
    #     uN = self.solve(sample, N, assemble)
    #
    #     pN = self.solveAdjoint(sample, N, assemble)
    #
    #     res_f = 0.
    #     for i in range(self.Qf_rb):
    #         res_f += coeff_f[i]*np.dot(self.fi_W[i][:N], pN)
    #
    #     res_a = 0.
    #     for i in range(self.Qa_rb):
    #         res_a += coeff_a[i]*np.dot(uN, np.dot(self.Vt_Ai_W[i][:N, :N], pN))
    #
    #     res = np.abs(res_f - res_a)
    #
    #     return res

    ################################ for misfit problem #################################################################

    # def coeffEvaluationMisfit(self,sample):
    #     """
    #     Evaluate the coefficients of the affine terms
    #     :param sample: np.random.uniform(-1,1,num_para)
    #     :return: coefficients
    #     """
    #     coeff_a, _ = self.HiFi.coeffEvaluation(sample)
    #
    #     coeff_f = [1.]
    #
    #     return coeff_a, coeff_f

    def solveAdj(self, sample, uN, N=0, assemble=True):

        coeff_a, coeff_f = self.coeffEvaluation(sample)

        # if hasattr(self, 'problem_type'):
        #     if self.problem_type == 'nonaffine' and assemble:
        #         # print "assemble", assemble
        #         # assemble HiFi matrix and vector and project them to RB matrix and vector
        #         self.HiFi.assemble(sample)
        #         self.At_rb, self.f_rb_misfit = SystemProjection(self.HiFi, self.basisAdj, self.projection_type)

        if N == 0:
            N = self.N

        if self.projection_type == 'Galerkin':
            A = coeff_a[0]*self.At_rb[0][:N, :N]
            for i in range(1, self.HiFi.Qa_rb):
                A += coeff_a[i]*self.At_rb[i][:N, :N]

            Bu_d = np.dot(self.B_V[:,:N], uN) - self.HiFi.misfit.d.get_local()
            f = -1./self.HiFi.misfit.noise_variance * np.dot(Bu_d.T, self.B_W[:,:N])

        elif self.projection_type == 'PetrovGalerkin':

            A = 0.*self.At_rb[0][:N, :N]
            f = 0.*self.f_rb_misfit[0][:N]
            # to be updated for misfit
            for i in range(self.HiFi.Qa_rb):
                for j in range(self.HiFi.Qa_rb):
                    A += coeff_a[i]*coeff_a[j]*self.At_rb[i*self.HiFi.Qa_rb+j][:N,:N]
                for j in range(self.HiFi.Qf_rb):
                    f += coeff_a[i]*coeff_f[j]*self.f_rb_misfit[i*self.HiFi.Qa_rb+j][:N]
            f *= -1.

        else:
            raise NameError('choose between Galerkin and PetrovGalerkin')

        pN = np.linalg.solve(A, f)

        return pN

    def reconstructAdj(self, sample, N=0, assemble=True):

        if N == 0:
            N = self.N

        uN = self.solve(sample, N, assemble)

        pN = self.solveAdj(sample, uN, N, assemble)

        p_rb = pN[0]*self.basisAdj[:,0]
        for i in range(1, N):
            p_rb += pN[i]*self.basisAdj[:,i]

        # # print "dofs",self.dofs
        solution = np.zeros(self.dofs)
        solution[self.node_inner] = p_rb

        # solution = u_rb
        # if N >= 90:
        # u_fun = dl.Function(self.HiFi.Vh[0])
        # u_fun.vector().set_local(solution)
        # dl.plot(u_fun)
        # dl.interactive()

        # hifi_solution = self.HiFi.solve(sample, self.reconstruct(sample))
        #
        # print("in reconstructMisfit error_du = ", np.linalg.norm(solution-hifi_solution, 2)/np.linalg.norm(hifi_solution,2))

        return solution

    def error_estimate_adj(self, sample, N=0, assemble=True):

        if N is 0:
            N = self.N

        coeff_a, coeff_f = self.coeffEvaluation(sample)
        # if self.HiFi.pde.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
        #     for i in range(self.Qa_rb):
        #         coeff_f.append(-coeff_a[i])

        uN = self.solve(sample, N, assemble)

        pN = self.solveAdj(sample, uN, N, assemble)

        res_f = 0.
        for i in range(self.Qf_rb):
            res_f += coeff_f[i]*np.dot(self.Wt_fi[i][:N], pN)

        res_a = 0.
        for i in range(self.Qa_rb):
            res_a += coeff_a[i]*np.dot(pN, np.dot(self.Wt_Ai_V[i][:N,:N], uN))

        res = res_a - res_f

        res /= self.cost(sample, N)  # relative error

        # # test accuracy
        # u = self.reconstruct(sample)
        # p = self.reconstructAdj(sample)
        # g = [u, sample, p]
        # [self.HiFi.x[i].set_local(g[i]) for i in range(3)]
        # import dolfin as dl
        # x = [dl.Function(self.HiFi.pde.Vh[i], self.HiFi.x[i]) for i in range(3)]
        # res_ass = dl.assemble(self.HiFi.pde.varf_handler(x[0],x[1],x[2]))
        # print("res - res_ass", res-res_ass)

        # # test accuracy
        # u_rb = self.reconstruct(sample)
        # # p_u_rb = self.HiFi.solveAdj(sample, u_rb)
        # p_u_rb = self.reconstructAdj(sample)
        # self.HiFi.uhelp.set_local(u_rb)
        # x = [self.HiFi.uhelp, None, None]
        # cost_rb = self.HiFi.misfit.cost(x)
        # self.HiFi.misfit.grad(0, x, self.HiFi.x[0])
        # grad_rb = self.HiFi.x[0].get_local()
        #
        # u = self.HiFi.solve(sample)
        # p = self.HiFi.solveAdj(sample, u)
        # self.HiFi.uhelp.set_local(u)
        # x = [self.HiFi.uhelp, None, None]
        # cost = self.HiFi.misfit.cost(x)
        # self.HiFi.misfit.grad(0, x, self.HiFi.x[0])
        # grad = self.HiFi.x[0].get_local()
        #
        # head = ["u-u_rb", "p-p_u_rb", "cost-cost_rb",  "dwr", "cost-cost_rb-dwr", "grad-grad_rb"]
        # print('{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}'.format(*head))
        # data = [np.sqrt(np.dot(np.dot(self.HiFi.Xnorm, u-u_rb), u-u_rb)),
        #         np.sqrt(np.dot(np.dot(self.HiFi.Xnorm, p - p_u_rb), p - p_u_rb)),
        #         np.abs(cost - cost_rb) / cost,
        #         np.abs(res) / cost,
        #         np.abs(cost - cost_rb - res) / cost,
        #         np.sqrt(np.dot(np.dot(self.HiFi.Xnorm, grad - grad_rb), grad - grad_rb)) / np.sqrt(
        #             np.dot(np.dot(self.HiFi.Xnorm, grad), grad))]
        #
        # print('{:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e} {:<20.2e}'.format(*data))

        res = np.abs(res)

        return res

    def cost(self, sample, N=0, assemble=True, dwr_modify=False):

        if N is 0:
            N = self.N

        coeff_a, coeff_f = self.coeffEvaluation(sample)

        uN = self.solve(sample, N, assemble)

        Bu_d = np.dot(self.B_V[:,:N], uN) - self.HiFi.misfit.d.get_local()
        cost = 0.5 / self.HiFi.misfit.noise_variance * np.dot(Bu_d, Bu_d)

        if dwr_modify:
            pN = self.solveAdj(sample, uN, N, assemble)

            res_f = 0.
            for i in range(self.Qf_rb):
                res_f += coeff_f[i] * np.dot(self.Wt_fi[i][:N], pN)

            res_a = 0.
            for i in range(self.Qa_rb):
                res_a += coeff_a[i] * np.dot(pN, np.dot(self.Wt_Ai_V[i][:N, :N], uN))

            res = res_a - res_f

            cost += res

        return cost

    def evalGradientParameter(self, sample, grad, N=0, assemble=True, dwr_modify=False):

        if N is 0:
            N = self.N

        grad.zero()
        coeff_a, coeff_f = self.coeffEvaluation(sample)
        coeff_grad_a, coeff_grad_f = self.coeffGradient(sample)

        uN = self.solve(sample, N, assemble)
        pN = self.solveAdj(sample, uN, N, assemble)
        tmp = self.HiFi.pde.generate_parameter()

        for i in range(self.Qa_rb):
            tmp.set_local(coeff_grad_a[i] * np.dot(pN, np.dot(self.Wt_Ai_V[i][:N, :N], uN)))
            grad.axpy(1.0, tmp)

        for i in range(self.Qf_rb):
            tmp.set_local(coeff_grad_f[i] * np.dot(pN, self.Wt_fi[i][:N]))
            grad.axpy(-1.0, tmp)

        # uN = self.solve(sample, N, assemble)
        # u = np.zeros(self.dofs)
        # u[self.node_inner] = np.dot(self.basis[:, :N], uN)
        # u[self.node_Dirichlet] = self.u_Dirichlet
        # u_vec = self.HiFi.pde.generate_state()
        # u_vec.set_local(u)
        #
        # pN = self.solveAdj(sample, uN, N, assemble)
        # p = np.zeros(self.dofs)
        # p[self.node_inner] = np.dot(self.basisAdj[:, :N], pN)
        # p_vec = self.HiFi.pde.generate_state()
        # p_vec.set_local(p)
        #
        # sample_vec = self.HiFi.pde.generate_parameter()
        # sample_vec.set_local(sample)
        # tmp = self.HiFi.pde.generate_parameter()
        #
        # x_u = [u_vec, sample_vec, p_vec]
        # self.HiFi.pde.evalGradientParameter(x_u, tmp)
        # grad.zero()
        # grad.axpy(1.0, tmp)

        if dwr_modify:
            # solve pNstar
            rhs = coeff_f[0] * self.Wt_fi[0][:N]
            for i in range(1, self.Qf_rb):
                rhs += coeff_f[i] * self.Wt_fi[i][:N]
            for i in range(self.Qa_rb):
                rhs -= coeff_a[i] * np.dot(self.Wt_Ai_V[i][:N,:N], uN)

            lhs = coeff_a[0] * self.At_rb[0][:N,:N]
            for i in range(1, self.Qa_rb):
                lhs += coeff_a[i] * self.At_rb[i][:N,:N]

            pNstar = np.linalg.solve(lhs, rhs)

            # solve uNstar
            rhs = -coeff_a[0] * np.dot(pN, self.Wt_Ai_V[0][:N,:N])
            for i in range(1, self.Qa_rb):
                rhs -= coeff_a[i] * np.dot(pN, self.Wt_Ai_V[i][:N,:N])
            Bu_d = np.dot(self.B_V[:,:N], uN) - self.HiFi.misfit.d.get_local()
            rhs -= 1. / self.HiFi.misfit.noise_variance * np.dot(Bu_d.T, self.B_V[:,:N])
            Bu_d = np.dot(self.B_W[:,:N], pNstar)
            rhs -= 1. / self.HiFi.misfit.noise_variance * np.dot(Bu_d.T, self.B_V[:,:N])

            lhs = coeff_a[0] * self.A_rb[0][:N,:N]
            for i in range(1, self.Qa_rb):
                lhs += coeff_a[i] * self.A_rb[i][:N,:N]

            uNstar = np.linalg.solve(lhs, rhs)

            # # evaluate gradient
            # ustar = np.zeros(self.dofs)
            # ustar[self.node_inner] = np.dot(self.basis[:,:N], uNstar)
            # ustar_vec = self.HiFi.x[0].copy()
            # ustar_vec.set_local(ustar)
            #
            # pstar = np.zeros(self.dofs)
            # pstar[self.node_inner] = np.dot(self.basisAdj[:,:N], pNstar)
            # pstar_vec = self.HiFi.x[0].copy()
            # pstar_vec.set_local(pstar)
            #
            # x_u = [u_vec, sample_vec, ustar_vec]
            # self.HiFi.pde.evalGradientParameter(x_u, tmp)
            # grad.axpy(1.0, tmp)

            for i in range(self.Qa_rb):
                tmp.set_local(coeff_grad_a[i] * np.dot(np.dot(uNstar, self.A_rb[i][:N, :N]), uN))
                grad.axpy(1.0, tmp)

            for i in range(self.Qf_rb):
                tmp.set_local(coeff_grad_f[i] * np.dot(uNstar, self.f_rb[i][:N]))
                grad.axpy(-1.0, tmp)

            # x_p = [pstar_vec, sample_vec, p_vec]
            # self.HiFi.pde.evalGradientParameter(x_p, tmp)
            # grad.axpy(1.0, tmp)

            for i in range(self.Qa_rb):
                tmp.set_local(coeff_grad_a[i] * np.dot(pN, np.dot(self.At_rb[i][:N, :N], pNstar)))
                grad.axpy(1.0, tmp)

        return grad