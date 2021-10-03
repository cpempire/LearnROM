import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../" )
# sys.path.append('../../../')
from hippylib import *

class PreconditionedHessianSVD:

    def __init__(self, pde, qoi, distribution, tol):
        self.pde = pde
        self.qoi = qoi
        self.distribution = distribution
        self.rhs_fwd = pde.generate_state()
        self.rhs_adj = pde.generate_state()
        self.rhs_adj2 = pde.generate_state()
        self.rhs_adj3 = pde.generate_state()
        self.mhelp = pde.generate_parameter()
        self.Hmhat1 = pde.generate_parameter()
        self.tol = tol
    # implement init_vector for Hessian in doublePassG
    def init_vector(self, m, dim):
        self.pde.init_parameter(m)

    # implement mult for Hessian action
    def mult(self, mhat1, Hmat1):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        # covariance preconditioned Hessian
        self.distribution.Rsolver.solve(self.mhelp, self.Hmhat1)
        # self.distribution.Rsolver.mult(self.Hmhat1, self.mhelp)

        Hmat1[:] = self.mhelp[:]

    def HessianInner(self, mhat1, mhat2):
        xhat = self.pde.generate_state()
        yhat = self.pde.generate_state()
        self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
        self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
        self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
        self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
        self.rhs_adj.axpy(1., self.rhs_adj2)
        self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
        self.rhs_adj.axpy(1., self.rhs_adj3)
        self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
        self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
        self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)
        self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
        self.Hmhat1.axpy(1., self.mhelp)

        # covariance preconditioned Hessian
        self.distribution.Rsolver.solve(self.mhelp, self.Hmhat1)

        return mhat2.inner(self.mhelp), xhat, yhat

class PreconditionedHessianAverageSVD:

    def __init__(self, pde, qoi, tol, distribution,samples=None):
        self.pde = pde
        self.qoi = qoi
        self.distribution = distribution
        self.samples = samples
        self.rhs_fwd = pde[0].generate_state()
        self.rhs_adj = pde[0].generate_state()
        self.rhs_adj2 = pde[0].generate_state()
        self.rhs_adj3 = pde[0].generate_state()
        self.mhelp = pde[0].generate_parameter()
        self.Hmhat1 = pde[0].generate_parameter()
        self.tol = tol
    # implement init_vector for Hessian in doublePassG
    def init_vector(self, m, dim):
        self.pde[0].init_parameter(m)

    # implement mult for Hessian action
    def mult(self, mhat1, Hmat1):

        for i in range(len(self.samples)):

            pde = self.pde[i]
            qoi = self.qoi[i]

            xhat = pde.generate_state()
            yhat = pde.generate_state()
            pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
            pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
            pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
            pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
            self.rhs_adj.axpy(1., self.rhs_adj2)
            qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
            self.rhs_adj.axpy(1., self.rhs_adj3)
            pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
            pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
            pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
            self.Hmhat1.axpy(1., self.mhelp)
            pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
            self.Hmhat1.axpy(1., self.mhelp)

            # covariance preconditioned Hessian
            self.distribution.Rsolver.solve(self.mhelp, self.Hmhat1)
            # self.distribution.Rsolver.mult(self.Hmhat1, self.mhelp)

            Hmat1[:] += self.mhelp[:]

        Hmat1[:] /= len(self.samples)

    # # implement mult for Hessian action
    # def mult(self, mhat1, Hmat1):
    #
    #     Hmat1[:] = self.Hmhat1[:]
    #     for i in range(self.nSamples):

            # u = self.pde.generate_state()
            # m = self.pde.generate_parameter()
            # m.set_local(self.samples[i,:])
            # p = self.pde.generate_state()
            # x_all = [u,m,p]
            # x = self.pde.generate_state()
            # self.pde.solveFwd(x,x_all,tol=1e-12)
            # x_all[STATE] = x
            # # solve the adjoint problem at the mean
            # y = self.pde.generate_state()
            # rhs = self.pde.generate_state()
            # self.qoi.adj_rhs(x,rhs)
            # self.pde.solveAdj(y,x_all,rhs,tol=1e-12)
            # x_all[ADJOINT] = y
            #
            # # set the linearization point for incremental forward and adjoint problems
            # self.pde.setLinearizationPoint(x_all)
            # self.qoi.setLinearizationPoint(x_all)

            # xhat = self.pde.generate_state()
            # yhat = self.pde.generate_state()
            # self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
            # self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
            # self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
            # self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
            # self.rhs_adj.axpy(1., self.rhs_adj2)
            # self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
            # self.rhs_adj.axpy(1., self.rhs_adj3)
            # self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
            # self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
            # self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
            # self.Hmhat1.axpy(1., self.mhelp)
            # self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
            # self.Hmhat1.axpy(1., self.mhelp)
            #
            # Hmat1[:] += self.Hmhat1[:]
    # def HessianInner(self, mhat1, mhat2):
    #     xhat = self.pde.generate_state()
    #     yhat = self.pde.generate_state()
    #     self.pde.apply_ij(ADJOINT,PARAMETER, mhat1, self.rhs_fwd)
    #     self.pde.solveIncremental(xhat, -self.rhs_fwd, False, self.tol) # False for forward,
    #     self.pde.apply_ij(STATE, STATE, xhat, self.rhs_adj)
    #     self.pde.apply_ij(STATE, PARAMETER, mhat1, self.rhs_adj2)
    #     self.rhs_adj.axpy(1., self.rhs_adj2)
    #     self.qoi.apply_ij(STATE,STATE,xhat,self.rhs_adj3)
    #     self.rhs_adj.axpy(1., self.rhs_adj3)
    #     self.pde.solveIncremental(yhat, -self.rhs_adj, True, self.tol) # True for adjoint,
    #     self.pde.apply_ij(PARAMETER,PARAMETER, mhat1, self.Hmhat1)
    #     self.pde.apply_ij(PARAMETER,ADJOINT,yhat, self.mhelp)
    #     self.Hmhat1.axpy(1., self.mhelp)
    #     self.pde.apply_ij(PARAMETER,STATE, xhat, self.mhelp)
    #     self.Hmhat1.axpy(1., self.mhelp)
    #
    #     return mhat2.inner(self.Hmhat1), xhat, yhat

