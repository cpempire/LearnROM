import dolfin as dl
import numpy as np
from scipy.sparse import csr_matrix
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../../" )
# sys.path.append( "../../../../" )
from hippylib import *

class HighFidelitySystem:

    def __init__(self, num_terms):
        """
        Initialize the High-fidelity system with 1. parameters; 2. mesh; 3. FE spaces;
        4. BC conditions; 5. HiFi mat, vec, Xnorm; 6. node_inner, node_Dirichlet, u_Dirichlet
        :param num_terms: 16, 64, etc.
        :return: HiFi class
        """
        self.problem_type = 'affine'

        # 1. parameters
        self.num_para  = num_terms
        self.para_mean = np.zeros(self.num_para)
        # self.para_min  = -np.sqrt(3)*np.ones(self.num_para)
        # self.para_max  = np.sqrt(3)*np.ones(self.num_para)
        self.para_min = -np.sqrt(3)*1e4*np.power(np.linspace(1.,self.num_para, self.num_para),-0.5)
        self.para_max = np.sqrt(3)*1e4*np.power(np.linspace(1.,self.num_para, self.num_para),-0.5)
        self.CovMat = np.diag(1e8*np.power(np.linspace(1.,self.num_para, self.num_para),-1))
        self.CovInv = np.diag(1e-8*np.power(np.linspace(1.,self.num_para, self.num_para),1))
        self.CovSqrt = np.diag(1e4*np.power(np.linspace(1.,self.num_para, self.num_para),-0.5))
        self.a0 = 5e4

        # 2. mesh construction
        nx = 64
        ny = 64
        mesh = dl.UnitSquareMesh(nx, ny)
        self.mesh = mesh

        # 3. finite element space
        Vh_STATE = dl.FunctionSpace(mesh,'CG',1)
        Vh_PARAMETER = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=num_terms)
        Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]
        self.Vh = Vh
        self.dofs = Vh_STATE.dofmap().dofs().size


        # for i in range(4):
        #     sample = dl.Function(Vh_PARAMETER).vector()
        #     parameter = np.random.uniform(-np.sqrt(3), np.sqrt(3), num_terms) + self.a0
        #     sample.set_local(parameter)
        #     sample_fun = vector2Function(sample,Vh[PARAMETER], name="sample")
        #
        #     fig = dl.plot(sample_fun)
        #     filename = 'figure/sample_'+str(i)+'.pvd'
        #     dl.File(filename) << sample_fun
        #     filename = 'figure/sample_'+str(i)+'.png'
        #     fig.write_png(filename)



        # 4. boundary conditions
        def boundary(x,on_boundary):
            return on_boundary  and (x[1] > 1.0 - dl.DOLFIN_EPS or x[1] < 0.0 + dl.DOLFIN_EPS) #or x[1] < 0.0 + dl.DOLFIN_EPS

        u_bdr = dl.Expression("1-x[1]", degree=1)
        bc = dl.DirichletBC(Vh_STATE, u_bdr, boundary)
        u_bdr0 = dl.Expression("0.0", degree=1)
        bc0 = dl.DirichletBC(Vh_STATE, u_bdr0, boundary)

        self.bc = bc
        self.bc0 = bc0

        # 5 extract the Dirichlet boundary conditions/index and the inner index
        u = dl.TrialFunction(Vh_STATE)
        v = dl.TestFunction(Vh_STATE)

        f = dl.Expression("1.0", degree=1)
        ftemp = dl.assemble(f*v*dl.dx)
        bc0.apply(ftemp)
        self.node_inner = np.nonzero(ftemp.get_local())[0]
        self.node_Dirichlet = np.setdiff1d(range(ftemp.local_size()),  self.node_inner)

        zero = dl.Expression("0.0", degree=1)
        ftemp = dl.assemble(zero*v*dl.dx)
        bc.apply(ftemp)
        self.u_Dirichlet = ftemp.get_local()[self.node_Dirichlet]

        # 6 assemble the high-fidelity system
        self.Qa = num_terms + 1
        self.Qf = 1

        k_block_1d = np.int(np.sqrt(num_terms))

        indicatorlist = []
        for n in range(num_terms):
            j = np.mod(n,k_block_1d)
            i = (n - j)/k_block_1d
            indicatorlist.append(dl.Expression("sin((i+1)*pi*x[0])*sin((j+1)*pi*x[1])", i = i, j = j, degree=1)) #/pow((n+1),0.5)
        indicatorlist = dl.as_vector(indicatorlist)
        self.indicatorlist = indicatorlist

        # 6.1 assemble the lhs matrix
        A_hifi = ()
        A_hifi_inner = ()

        for n in range(self.Qa):
            print "# A_hifi_inner ", n,"/",self.Qa
            if n == 0:
                Atemp = dl.Constant(self.a0)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(v))*dl.dx
            else:
                Atemp = indicatorlist[n-1]*dl.inner(dl.nabla_grad(u), dl.nabla_grad(v))*dl.dx

            Atemp = dl.assemble(Atemp)
            A_hifi += (Atemp,)
            # A_hifi_inner += (Atemp.array()[np.ix_(self.node_inner, self.node_inner)],)

            # print "Atemp.array()[np.ix_(self.node_inner, self.node_inner)]", \
            #     Atemp.array()[np.ix_(self.node_inner, self.node_inner)].nbytes
            # print "csr_matrix(Atemp.array()[np.ix_(self.node_inner, self.node_inner)])",\
            #     csr_matrix(Atemp.array()[np.ix_(self.node_inner, self.node_inner)]).data.nbytes
            A_hifi_inner += (csr_matrix(Atemp.array()[np.ix_(self.node_inner, self.node_inner)]),)

        # 6.2 assemble the rhs vector
        f_hifi = ()
        f_hifi_inner = ()

        f = dl.Expression("0.0", degree=1)
        g = dl.Expression("-(1.0-x[1])*(x[0]>0.0)*(x[0]<1.0)", degree=1)
        ftemp = f*v*dl.dx #- g*v*dl.ds
        ftemp = dl.assemble(ftemp)
        f_hifi += (ftemp,)
        f_hifi_inner += (ftemp.get_local()[self.node_inner],)

        self.f = f
        self.g = g
        self.A_hifi = A_hifi
        self.A_hifi_inner = A_hifi_inner
        self.f_hifi = f_hifi
        self.f_hifi_inner = f_hifi_inner

        # lift the Dirichlet BC for reduced basis system
        self.Qa_rb = self.Qa
        self.Qf_rb = self.Qf
        if self.bc is not None and np.linalg.norm(self.u_Dirichlet,np.inf) > 2e-16:
            for i in range(self.Qa):
                ftemp_inner = self.A_hifi[i].array()[np.ix_(self.node_inner, self.node_Dirichlet)].dot(self.u_Dirichlet)
                self.f_hifi_inner += (ftemp_inner,)
            self.Qf_rb += self.Qa

        # 6.3 compute Xnorm
        coeff_a, _ = self.coeffEvaluation(self.para_mean)
        Xnorm = coeff_a[0]*self.A_hifi[0]
        for i in range(1,self.Qa):
            Xnorm += coeff_a[i]*self.A_hifi[i]
        self.Xnorm = Xnorm.array()
        # self.Xnorm_inner = self.Xnorm[np.ix_(self.node_inner, self.node_inner)]
        self.Xnorm_inner = csr_matrix(self.Xnorm[np.ix_(self.node_inner, self.node_inner)])
        self.M = dl.assemble(u*v*dl.dx)
        # self.M_inner = self.M.array()[np.ix_(self.node_inner, self.node_inner)]
        self.M_inner = csr_matrix(self.M.array()[np.ix_(self.node_inner, self.node_inner)])

    def coeffEvaluation(self,sample):
        """
        Evaluate the coefficients of the affine terms
        :param sample: np.random.uniform(-1,1,num_para)
        :return: coefficients
        """
        coeff_a = np.zeros(self.Qa)

        coeff_a[0] = 1
        coeff_a[1:self.Qa] = sample

        # for i in range(self.Qa):
        #     # coeff_a[i] = np.power(10, (i+1)**(-1)*sample[i])
        #     coeff_a[i] = np.power(10, sample[i])

        coeff_f = [1.]

        return coeff_a, coeff_f


    def residual(self,u,m,p):
        """
        The residual in weak form of PDE problem, used for adjoint problem, Hessian action, etcs.
        :param u: state
        :param m: parameter
        :param p: adjoint
        :return: residual
        """

        # res = self.indicatorlist[0]*pow(10,m[0])*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - self.f*p*dl.dx
        # for i in range(1,self.num_para):
        #     # res += self.indicatorlist[i]*pow(10,(i+1)**(-1)*m[i])*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx
        #     res += self.indicatorlist[i]*pow(10,m[i])*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx

        res = (dl.Constant(self.a0)+dl.inner(self.indicatorlist,m))*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx \
              - self.f*p*dl.dx #+ self.g*p*dl.ds

        return res

    # def solve(self, sample):
    #     u = dl.TrialFunction(self.Vh[0])
    #     v = dl.TestFunction(self.Vh[0])
    #     m = dl.Function(self.Vh[1])
    #     m.vector().set_local(sample)
    #     res = self.residual(u,m,v)
    #
    #     A_form, f_form = dl.lhs(res), dl.rhs(res)
    #     A = dl.assemble(A_form)
    #     self.bc.apply(A)
    #     f = dl.assemble(f_form)
    #     self.bc.apply(f)
    #
    #     u = dl.Function(self.Vh[0])
    #
    #     dl.solve(A, u.vector(), f)
    #
    #     solution = u.vector().get_local()
    #
    #     return solution

    def solve(self, sample):
        """
        Solve the high-fidelity system
        :param sample: np.random.uniform(-1,1,num_para)
        :return: solution of np.ndarray type
        """
        coeff_a, coeff_f = self.coeffEvaluation(sample)

        # A = self.A_hifi[0]
        # for i in range(self.Qa):
        #     A += coeff_a[i]*self.A_hifi[i+1]
        #
        # f = self.f_hifi[0]
        # for i in range(self.Qf):
        #     f += coeff_f[i]*self.f_hifi[i+1]

        A = coeff_a[0]*self.A_hifi[0]
        for i in range(1, self.Qa):
            A += coeff_a[i]*self.A_hifi[i]
        self.bc.apply(A)

        f = coeff_f[0]*self.f_hifi[0]
        for i in range(1,self.Qf):
            f += coeff_f[i]*self.f_hifi[i]
        self.bc.apply(f)

        u = dl.Function(self.Vh[0])

        dl.solve(A, u.vector(), f)

        solution = u.vector().get_local()

        # fig = dl.plot(u)
        # fig.write_png('figure/snapshot1.png')
        # dl.interactive()

        return solution

    def snapshots(self,samples):
        """
        Compute solutions at many samples
        :param samples: many samples
        :return: snapshots
        """
        num_samples = samples.shape[0]

        U = []
        for i in range(num_samples):
            sample = samples[i, :]
            u = self.solve(sample)
            U.append(u)
        U = np.array(U).T

        return U



class HighFidelitySystemAdjoint(HighFidelitySystem):

    def __init__(self, num_terms, qoi):
        """
        Inherite from HighFidelitySystem
        """
        HighFidelitySystem.__init__(self, num_terms)

        self.qoi = qoi

        self.bc = self.bc0
        self.u_Dirichlet = 0.*self.u_Dirichlet

        # assemble the rhs vector
        f_hifi = ()
        f_hifi_inner = ()

        f_hifi += (self.qoi.qoi,)
        f_hifi_inner += (self.qoi.qoi.get_local()[self.node_inner],)

        self.f_hifi = f_hifi
        self.f_hifi_inner = f_hifi_inner

        # lift the Dirichlet BC for reduced basis system
        self.Qa_rb = self.Qa
        self.Qf_rb = self.Qf
