import dolfin as dl
import numpy as np

class NonLinearReducedBasisSolver:
    """
    Newton-type non linear solver with backtracking line-search.
    Parameters for the nonlinear solver are set through the attribute parameter.
    - "rel_tolerance": the relative tolerance (1e-6 default)
    - "abs_tolerance": the absolute tolerance (1e-12 default)
    - "max_iter": the maximum number of iterations (200 default)
    - "print_level": controls verbosity of the solver to screen
                     -1: no output to screen
                      0: final residual, and convergence reason
                      1: print current residual and alpha at each iteration
    - "max_backtracking": maximum number of backtracking iterations
    """
    def __init__(self, problem, basis, Wr=None, callback=None, extra_args=()):
        """
        Constructor

        INPUT:
        - problem: an object of type NonlinearProblem. It provides the residual and the Jacobian
        - Wr: an s.p.d. operator used to compute a weighted norm.
            This object must provide the methods:
            Wr.norm(r):= sqrt( r, Wr*r) to compute the weighted norm of r
            Wr.inner(r1,r2) := (r1, Wr*r2) to compute the weighted inner product of r1 and r2
            If Wr is None the standard l2-norm is used
        - callback: a function handler to perform additional postprocessing (such as output to paraview)
                    of the current solution
        - extra_args: a tuple of additional parameters to evaluate the residual and Jacobian
        """
        self.parameters = {}
        self.parameters["rel_tolerance"]         = 1e-10
        self.parameters["abs_tolerance"]         = 1e-12
        self.parameters["max_iter"]              = 200
        self.parameters["print_level"]           = 0
        self.parameters["max_backtracking"]      = 20

        self.problem = problem
        self.basis = basis

        self.DEIM = None

        self.Wr = Wr

        self.callback = callback
        self.extra_args = extra_args

        self.final_it = 0
        self.final_norm = 0
        self.initial_norm = 0
        self.converged = False

    def _norm_r(self,r):
        if self.Wr is None:
            return np.sqrt(r.dot(r))
        else:
            return self.Wr.norm(r)

    def _inner_r(self,r1,r2):
        if self.Wr is None:
            return r1.dot(r2)
        else:
            return self.Wr.inner(r1,r2)

    def solve(self, HiFi, x, N, M=None):
        r_record = ()
        d_record = ()

        basis = self.basis[:,:N]

        if M != None:
            nodes = self.DEIM.nodes[:M]
            basis_deim = self.DEIM.bases[:,:M]

        rtol = self.parameters["rel_tolerance"]
        atol = self.parameters["abs_tolerance"]
        max_iter = self.parameters["max_iter"]
        max_backtracking = self.parameters["max_backtracking"]
        x_hifi = dl.Function(self.problem.Vh).vector()

        xtemp = np.zeros(HiFi.dofs)
        xtemp[HiFi.node_inner] = np.dot(basis,x)
        xtemp[HiFi.node_Dirichlet] = HiFi.u_Dirichlet

        x_hifi.set_local(xtemp)

        self.problem.applyBC(x_hifi)

        if self.callback is not None:
            self.callback(x_hifi)

        r_hifi = self.problem.residual(x_hifi, self.extra_args)
        r_temp = r_hifi.array()[HiFi.node_inner]
        r_record += (r_temp,)

        # apply hyper reduction
        if M != None:
            basisTbasis_deim = np.dot(basis.T,basis_deim)
            r_deim = np.linalg.solve(basis_deim[nodes,:], r_temp[nodes])
            r = np.dot(basisTbasis_deim, r_deim)
        else:
            r = np.dot(basis.T,r_temp)

        norm_r0 = self._norm_r(r)
        norm_r = norm_r0

        self.initial_norm = norm_r0

        tol = max(atol, rtol*norm_r0)

        self.converged = False
        it = 0

        if(self.parameters["print_level"] >= 1):
                print "\n{0:3} {1:15} {2:15}".format(
                      "It", "||r||", "alpha")

        if self.parameters["print_level"] >= 1:
                print "{0:3d} {1:15e} {2:15e}".format(
                        it, norm_r0, 1.)

        while it < max_iter and not self.converged:

            J_hifi = self.problem.Jacobian(x_hifi, self.extra_args)
            J_hifi = J_hifi.array()[np.ix_(HiFi.node_inner, HiFi.node_inner)]

            if M != None:
                # print "J_hifi[nodes,:].shape,basis.shape", J_hifi[nodes,:].shape,basis.shape
                J_temp = np.dot(J_hifi[nodes,:], basis)
                # print "basis_deim[nodes,:]", basis_deim[nodes,:].shape, J_temp.shape, basis_deim[nodes,:], J_temp[:].shape
                J_temp = np.linalg.solve(basis_deim[nodes,:],J_temp)
                J = np.dot(basisTbasis_deim, J_temp)

            else:
                J = np.dot(basis.T, np.dot(J_hifi, basis))

            d = np.linalg.solve(J, -r)
            d_hifi = np.dot(basis,d)

            d_record += (d_hifi,)

            alpha = 1.
            backtracking_converged = False
            j = 0
            x_new_hifi = dl.Function(self.problem.Vh).vector()

            while j < max_backtracking and not backtracking_converged:
                x_new = x + alpha*d

                xtemp = np.zeros(HiFi.dofs)
                xtemp[HiFi.node_inner] = np.dot(basis,x_new)
                xtemp[HiFi.node_Dirichlet] = HiFi.u_Dirichlet

                x_new_hifi.set_local(xtemp)

                r_hifi = self.problem.residual(x_new_hifi, self.extra_args)
                r_temp = r_hifi.array()[HiFi.node_inner]

                # apply hyper reduction
                if M != None:
                    r_deim = np.linalg.solve(basis_deim[nodes,:], r_temp[nodes])
                    r = np.dot(basisTbasis_deim, r_deim)
                else:
                    r = np.dot(basis.T,r_temp)

                try:
                    norm_r_new = self._norm_r(r)
                except:
                    norm_r_new = norm_r+1.
                if norm_r_new  <= norm_r:
                    x += alpha*d

                    xtemp = np.zeros(HiFi.dofs)
                    xtemp[HiFi.node_inner] = np.dot(basis,x)
                    xtemp[HiFi.node_Dirichlet] = HiFi.u_Dirichlet

                    x_hifi.set_local(xtemp)

                    norm_r = norm_r_new
                    backtracking_converged = True
                else:
                    alpha = .5*alpha
                    j = j+1

            r_record += (r_temp,)

            if not backtracking_converged:
                if self.parameters["print_level"] >= 0:
                    print "Backtracking failed at iteration", it, ". Residual norm is ", norm_r
                self.converged = False
                self.final_it = it
                self.final_norm = norm_r
                break

            if norm_r_new < tol:
                if self.parameters["print_level"] >= 0:
                    print "Converged in ", it, "iterations with final residual norm", norm_r_new
                self.final_norm = norm_r_new
                self.converged = True
                self.final_it = it
                break

            it = it+1

            if self.parameters["print_level"] >= 1:
                print "{0:3d} {1:15e} {2:15e}".format(
                        it,  norm_r, alpha)

            if self.callback is not None:
                self.callback(x_hifi)

        if not self.converged:
            self.final_norm = norm_r_new
            self.final_it = it
            if self.parameters["print_level"] >= 0:
                    print "Not Converged in ", it, "iterations. Final residual norm", norm_r_new

        return x_hifi.array(), d_record, r_record