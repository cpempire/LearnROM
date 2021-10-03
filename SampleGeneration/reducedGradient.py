import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append( dir_path+"/../../" )
# sys.path.append('../../../')
from hippylib import *

class ReducedGradient:

    def __init__(self, pde, qoi, tol, samples=None):
        self.pde = pde
        self.qoi = qoi
        self.tol = tol
        self.samples = samples
        self.nSamples = len(self.samples)

        self.Gradient = []
        for i in range(self.nSamples):

            # solve the forward problem at the m
            u = pde.generate_state()
            m = pde.generate_parameter()
            m.set_local(samples[i])
            p = pde.generate_state()
            x_all = [u,m,p]
            x = pde.generate_state()
            pde.solveFwd(x,x_all,tol=tol)
            x_all[STATE] = x
            # solve the adjoint problem at the m
            y = pde.generate_state()
            rhs = pde.generate_state()
            qoi.adj_rhs(x_all,rhs)
            pde.solveAdj(y,x_all,rhs,tol=tol)
            x_all[ADJOINT] = y

            # compute the gradient
            grad = pde.generate_parameter()
            pde.evalGradientParameter(x_all, grad)
            self.Gradient.append(grad)

    # implement init_vector for Hessian in doublePassG
    def init_vector(self, m, dim):
        self.pde.init_parameter(m)

    # implement mult for Hessian action
    def mult(self, mhat, Gmat):

        Gmat.zero()
        for i in range(self.nSamples):
            Gmat.axpy(self.Gradient[i].inner(mhat)/self.nSamples, self.Gradient[i])

class PreconditionedReducedGradient:

    def __init__(self, pde, qoi, prior,tol, samples=None):
        self.pde = pde
        self.qoi = qoi
        self.prior = prior
        self.tol = tol
        self.samples = samples
        self.nSamples = len(self.samples)

        self.Gradient = []
        for i in range(self.nSamples):

            # solve the forward problem at the m
            u = pde.generate_state()
            m = pde.generate_parameter()
            m.set_local(samples[i])
            p = pde.generate_state()
            x_all = [u,m,p]
            x = pde.generate_state()
            pde.solveFwd(x,x_all,tol=tol)
            x_all[STATE] = x
            # solve the adjoint problem at the m
            y = pde.generate_state()
            rhs = pde.generate_state()
            qoi.adj_rhs(x_all,rhs)
            pde.solveAdj(y,x_all,rhs,tol=tol)
            x_all[ADJOINT] = y

            # compute the gradient
            grad = pde.generate_parameter()
            pde.evalGradientParameter(x_all, grad)
            self.Gradient.append(grad)

    # implement init_vector for Hessian in doublePassG
    def init_vector(self, m, dim):
        self.pde.init_parameter(m)

    # implement mult for Hessian action
    def mult(self, mhat, CGmat):

        Gmat = self.pde.generate_parameter()
        for i in range(self.nSamples):
            Gmat.axpy(self.Gradient[i].inner(mhat)/self.nSamples, self.Gradient[i])
        self.prior.Rsolver.mult(Gmat, CGmat)
