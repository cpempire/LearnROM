import numpy as np
from scipy.sparse import csr_matrix

def SystemProjection(RB, basis):
    "project the high-fidelity system to the reduced basis system"
    A_rb = ()
    f_rb = ()

    HiFi = RB.HiFi
    projection_type = RB.projection_type

    if projection_type is "Galerkin":

        if HiFi.problem_type is 'nonaffine':
            HiFi.A_hifi_inner = ()
            for j in range(HiFi.Qa):
                HiFi.A_hifi_inner += (HiFi.A_hifi[j].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)],)

            HiFi.f_hifi_inner = ()
            for j in range(HiFi.Qf_rb):
                HiFi.f_hifi_inner += (HiFi.f_hifi[j].get_local()[HiFi.node_inner], )

        for i in range(HiFi.Qa_rb):
            # A_hifi_i = HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)]
            # print "A_hifi_i, basis", A_hifi_i.shape, basis.shape
            # Atemp = np.dot(basis.T, A_hifi_i.dot(basis))
            # print "basis, HiFi.A_hifi_inner[i]", basis.shape, HiFi.A_hifi_inner[i].shape
            Atemp = np.dot(basis.T, HiFi.A_hifi_inner[i].dot(basis))
            # if basis.shape[1] == 1:
            #     Atemp = np.array([[Atemp]])
            A_rb += (Atemp, )

        for i in range(HiFi.Qf_rb):
            # f_hifi_i = HiFi.f_hifi[i].get_local()[HiFi.node_inner]
            # ftemp = np.dot(basis.T, f_hifi_i)
            ftemp = np.dot(basis.T, HiFi.f_hifi_inner[i])
            # if basis.shape[1] == 1:
            #     ftemp = np.array([[ftemp]])
            f_rb += (ftemp, )

        # if HiFi.bc is not None and np.linalg.norm(HiFi.u_Dirichlet,np.inf) > 2e-16:
        #     for i in range(HiFi.Qa):
        #         f_hifi_i = HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_Dirichlet)].dot(HiFi.u_Dirichlet)
        #         ftemp = np.dot(basis.T, f_hifi_i)
        #         # if basis.shape[1] == 1:
        #         #     ftemp = np.array([[ftemp]])
        #         f_rb += (ftemp, )


    elif projection_type is "PetrovGalerkin":

        for i in range(HiFi.Qa_rb):
            for j in range(HiFi.Qa_rb):
                # MinvA = np.linalg.solve(HiFi.M.array()[np.ix_(HiFi.node_inner, HiFi.node_inner)],
                #                         HiFi.A_hifi[j].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)])
                # A_hifi_ij = np.dot(HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)].T,MinvA)
                # A_hifi_ij = np.dot(HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)].T,
                #                    HiFi.A_hifi[j].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)])

                A_hifi_ij = np.dot(HiFi.A_hifi_inner[i].T, np.linalg.solve(HiFi.M_inner, HiFi.A_hifi_inner[j]))
                Atemp = np.dot(basis.T, A_hifi_ij.dot(basis))
                A_rb += (Atemp, )

            for j in range(HiFi.Qf_rb):
                # Minvf = np.linalg.solve(HiFi.M.array()[np.ix_(HiFi.node_inner, HiFi.node_inner)],
                #                         HiFi.f_hifi[j].get_local()[HiFi.node_inner])
                # f_hifi_ij = np.dot(HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)].T, Minvf)
                # f_hifi_ij = np.dot(HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)].T,
                #                    HiFi.f_hifi[j].get_local()[HiFi.node_inner])
                f_hifi_ij = np.dot(HiFi.A_hifi_inner[i].T, np.linalg.solve(HiFi.M_inner,HiFi.f_hifi_inner[j]))
                ftemp = np.dot(basis.T, f_hifi_ij)
                f_rb += (ftemp, )

            # if HiFi.bc is not None and np.linalg.norm(HiFi.u_Dirichlet,np.inf) > 2e-16:
            #     for j in range(HiFi.Qa):
            #         f_hifi_j = HiFi.A_hifi[j].array()[np.ix_(HiFi.node_inner, HiFi.node_Dirichlet)].dot(HiFi.u_Dirichlet)
            #         Minvf = np.linalg.solve(HiFi.M.array()[np.ix_(HiFi.node_inner, HiFi.node_inner)], f_hifi_j)
            #         f_hifi_ij = np.dot(HiFi.A_hifi[i].array()[np.ix_(HiFi.node_inner, HiFi.node_inner)].T, Minvf)
            #         ftemp = np.dot(basis.T, f_hifi_ij)
            #         f_rb += (ftemp, )
    else:
        raise("projection_type is Galerkin or PetrovGalerkin")

    return A_rb, f_rb


def SystemProjectionAdjoint(RB, basis, basisAdj):
    "project the high-fidelity system to the reduced basis system"
    A_rb = ()
    f_rb = ()
    At_rb = ()
    HiFi = RB.HiFi
    projection_type = RB.projection_type

    if projection_type is "Galerkin":

        for i in range(HiFi.Qa_rb):
            Atemp = np.dot(basis.T, HiFi.A_hifi_inner[i].dot(basis))
            A_rb += (Atemp,)

        for i in range(HiFi.Qa_rb):
            Atemp = np.dot(basisAdj.T, HiFi.At_hifi_inner[i].dot(basisAdj))
            At_rb += (Atemp,)

        for i in range(HiFi.Qf_rb):
            ftemp = np.dot(basis.T, HiFi.f_hifi_inner[i])
            f_rb += (ftemp,)

        # gradient of misfit term, B_V = (B * basis), B_W = (B * basisAdj), matrix of (s x N)
        B_V = np.zeros([len(HiFi.misfit.d.get_local()), RB.N])
        B_W = np.zeros([len(HiFi.misfit.d.get_local()), RB.N])
        for n in range(RB.N):
            HiFi.uhelp_array[HiFi.node_inner] = basis[:,n]
            HiFi.uhelp.set_local(HiFi.uhelp_array)
            HiFi.misfit.B.mult(HiFi.uhelp, HiFi.misfit.help)
            B_V[:, n] = HiFi.misfit.help.get_local()

            HiFi.uhelp_array[HiFi.node_inner] = basisAdj[:,n]
            HiFi.uhelp.set_local(HiFi.uhelp_array)
            HiFi.misfit.B.mult(HiFi.uhelp, HiFi.misfit.help)
            B_W[:, n] = HiFi.misfit.help.get_local()

    elif projection_type is "PetrovGalerkin":

        raise ("to be implemented")

    else:
        raise ("projection_type is Galerkin or PetrovGalerkin")

    return A_rb, f_rb, At_rb, B_V, B_W

# class SystemProjection:
#     """Project the system by Galerkin or PetrovGalerkin projection_type"""
#     def __init__(self, HiFi, basis, projection_type):
#         """
#         :param HiFi: HiFi class
#         :param basis: basis functions
#         :param projection_type: Galerkin, PetrovGalerkin
#         :return: A_rb, f_rb
#         """
#         self.HiFi = HiFi
#         self.basis = basis
#         self.projection_type = projection_type
#
#         if projection_type == 'Galerkin':
#             self.A_rb, self.f_rb = self.GalerkinProjection()
#         elif projection_type == 'PetrovGalerkin': # Least Squares PG
#             self.A_rb, self.f_rb = self.PetrovGalerkinProjection()
#
#     def GalerkinProjection(self):
#         A_rb = ()
#         for i in range(self.HiFi.Qa):
#             A_hifi_i = self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)]
#             # print "A_hifi_i, basis", A_hifi_i.shape, self.basis.shape
#             Atemp = np.dot(self.basis.T, A_hifi_i.dot(self.basis))
#             A_rb += (Atemp, )
#
#         f_rb = ()
#         for i in range(self.HiFi.Qf):
#             f_hifi_i = self.HiFi.f_hifi[i].get_local()[self.HiFi.node_inner]
#             ftemp = np.dot(self.basis.T, f_hifi_i)
#             f_rb += (ftemp, )
#
#         if self.HiFi.bc is not None and np.linalg.norm(self.HiFi.u_Dirichlet,np.inf) > 2e-16:
#             for i in range(self.HiFi.Qa):
#                 f_hifi_i = self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_Dirichlet)].dot(self.HiFi.u_Dirichlet)
#                 ftemp = np.dot(self.basis.T, f_hifi_i)
#                 f_rb += (ftemp, )
#
#         return A_rb, f_rb
#
#     def PetrovGalerkinProjection(self):
#         A_rb = ()
#         f_rb = ()
#         for i in range(self.HiFi.Qa):
#             for j in range(self.HiFi.Qa):
#                 MinvA = np.linalg.solve(self.HiFi.M.array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)],
#                                         self.HiFi.A_hifi[j].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)])
#                 A_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T,MinvA)
#                 # A_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T,
#                 #                    self.HiFi.A_hifi[j].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)])
#                 Atemp = np.dot(self.basis.T, A_hifi_ij.dot(self.basis))
#                 A_rb += (Atemp, )
#
#             for j in range(self.HiFi.Qf):
#                 Minvf = np.linalg.solve(self.HiFi.M.array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)],
#                                         self.HiFi.f_hifi[j].get_local()[self.HiFi.node_inner])
#                 f_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T, Minvf)
#                 # f_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T,
#                 #                    self.HiFi.f_hifi[j].get_local()[self.HiFi.node_inner])
#                 ftemp = np.dot(self.basis.T, f_hifi_ij)
#                 f_rb += (ftemp, )
#
#             if self.HiFi.bc is not None and np.linalg.norm(self.HiFi.u_Dirichlet,np.inf) > 2e-16:
#                 for j in range(self.HiFi.Qa):
#                     f_hifi_j = self.HiFi.A_hifi[j].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_Dirichlet)].dot(self.HiFi.u_Dirichlet)
#                     Minvf = np.linalg.solve(self.HiFi.M.array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)], f_hifi_j)
#                     f_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T, Minvf)
#                     ftemp = np.dot(self.basis.T, f_hifi_ij)
#                     f_rb += (ftemp, )
#
#             # if self.HiFi.bc is not None:
#             #     print "to be implemented"
#
#         return A_rb, f_rb