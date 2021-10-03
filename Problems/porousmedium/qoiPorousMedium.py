import dolfin as dl


class QoIPorousMedium:

    def __init__(self, mesh, Vh_STATE):
        """
        Constructor.
        INPUTS:
        - mesh: the mesh
        - Vh_STATE: the finite element space for the state variable
        """
        chi = dl.Expression("(x[0]<=0.1)*(x[1]<=0.1)/0.01", degree=1)
        # chi = dl.Expression("(x[0]>=0.9)*(x[1]>=0.9)/0.01", degree=1)
        # chi = dl.Expression("(x[0]>=0.45)*(x[0]<=0.55)*(x[1]>=0.45)*(x[1]<=0.55)/0.01", degree=1)

        # chi = dl.Constant(1.0)
        v = dl.TestFunction(Vh_STATE)
        self.qoi = dl.assemble(v*chi*dl.dx)

        self.state = dl.Function(Vh_STATE).vector()
        self.help = dl.Function(Vh_STATE).vector()
        self.Vh_STATE = Vh_STATE

    def eval(self, x):
        """
        Evaluate the quantity of interest at a given point in the state and
        parameter space.

        INPUTS:
        - x coefficient vector of state variable
        """
        QoI = self.qoi.inner(x[0])
        return QoI

    def adj_rhs(self,x,rhs):
        """
        The right hand for the adjoint problem (i.e. the derivative of the Lagrangian funtional
        with respect to the state u).

        INPUTS:
        - x coefficient vector of state variable
        - rhs: FEniCS vector to store the rhs for the adjoint problem.
        """
        ### rhs = - df/dstate
        self.grad_state(x, rhs)
        rhs *= -1

    def grad_state(self,x,g):
        """
        The partial derivative of the qoi with respect to the state variable.

        INPUTS:
        - x coefficient vector of state variable
        - g: FEniCS vector to store the gradient w.r.t. the state.
        """
        g.zero()
        g.axpy(1.,self.qoi)

    def apply_ij(self,i,j, dir, out):
        """
        Apply the second variation \delta_ij (i,j = STATE,PARAMETER) of the q.o.i. in direction dir.

        INPUTS:
        - i,j integer (STATE=0, PARAMETER=1) which indicates with respect to which variables differentiate
        - dir the direction in which to apply the second variation
        - out: FEniCS vector to store the second variation in the direction dir.

        NOTE: setLinearizationPoint must be called before calling this method.
        """

        out.zero()

    def apply_ijk(self,i,j,k,dir1,dir2, out):
        ## Q_xxx(dir1, dir2, x_test)

        out.zero()

    def setLinearizationPoint(self, x):
        """
        Specify the linearization point for computation of the second variations in method apply_ij.

        INPUTS:
        - x = [u,m,p] is a list of the state u, parameter m, and adjoint variable p
        """
        self.state.zero()
        self.state.axpy(1., x[0])
