
import numpy as np
from .systemProjection import SystemProjection, SystemProjectionAdjoint
from .residualProjection import ResidualProjection, ResidualProjectionAdjoint
from .gramSchmidt import GramSchmidt
import time

def BuildReducedBasisGreedy(RB, samples, tol, Nmax):

    RB.N = 0
    nSamples = len(samples)

    samples_active_set = np.ones(nSamples)
    sample_index = 0
    sample_index_set =[0,]
    samples_active_set[sample_index] = 0
    error_max_set = [1,]
    error_max = tol + 1.
    basis = []
    if RB.HiFi.misfit is not None:
        basisAdj = []

    while RB.N < Nmax and error_max > tol:

        RB.N += 1

        # solve the high fidelity system
        solution = RB.HiFi.solve(samples[sample_index,:])
        if RB.HiFi.misfit is not None:
            solutionAdj = RB.HiFi.solveAdj(samples[sample_index,:], solution)

        # Gram Schmidt orthogonalization
        solution_orthogonal = GramSchmidt(basis, solution[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)
        if RB.HiFi.misfit is not None:
            solutionAdj_orthogonal = GramSchmidt(basisAdj, solutionAdj[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)

        # expand the basis
        if len(basis) > 0:
            basis = np.hstack((basis, np.array([solution_orthogonal]).T))
        else:
            basis = np.array([solution_orthogonal]).T
        RB.basis = basis
        if RB.HiFi.misfit is not None:
            if len(basisAdj) > 0:
                basisAdj = np.hstack((basisAdj, np.array([solutionAdj_orthogonal]).T))
            else:
                basisAdj = np.array([solutionAdj_orthogonal]).T
            RB.basisAdj = basisAdj

        if RB.HiFi.misfit is not None:
            RB.A_rb, RB.f_rb, RB.At_rb, RB.B_V, RB.B_W = SystemProjectionAdjoint(RB, basis, basisAdj)
        else:
            RB.A_rb, RB.f_rb = SystemProjection(RB, basis)

        if RB.HiFi.misfit is not None:
            RB.Wt_fi, RB.Wt_Ai_V = ResidualProjectionAdjoint(RB.HiFi, basis, basisAdj)
        else:
            RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(RB.HiFi, basis)

        error_estimate = np.zeros(nSamples)
        for i in range(nSamples):
            if samples_active_set[i]:
                if RB.HiFi.misfit is not None:
                    error_estimate[i] = RB.error_estimate_adj(samples[i,:], RB.N, assemble=True)
                else:
                    error_estimate[i] = RB.error_estimate(samples[i,:], RB.N, assemble=True)

        error_max = np.amax(error_estimate)
        sample_index = np.argmax(error_estimate)
        sample_index_set.append(sample_index)
        error_max_set.append(error_max)
        samples_active_set[sample_index] = 0

        # print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)

    RB.error_max_set = error_max_set
    RB.sample_index_set = sample_index_set


def UpdateReducedBasisGreedy(RB, samples, restart=False):

    nSamples = len(samples)

    samples_active_set = np.ones(nSamples)
    sample_index = 0
    sample_index_set =[0,]
    samples_active_set[sample_index] = 0
    error_max_set = [1,]
    error_max = RB.tol + 1.
    if restart:
        RB.N = 1
        basis = RB.basis[:,:1]
        if RB.HiFi.misfit is not None:
            basisAdj = RB.basisAdj[:,:1]
    else:
        basis = RB.basis
        if RB.HiFi.misfit is not None:
            basisAdj = RB.basisAdj

    while RB.N < RB.Nmax and error_max > RB.tol:

        error_estimate = np.zeros(nSamples)

        for i in range(nSamples):
            if samples_active_set[i]:
                if RB.HiFi.misfit is not None:
                    error_estimate[i] = RB.error_estimate_adj(samples[i,:], RB.N, assemble=True)
                else:
                    error_estimate[i] = RB.error_estimate(samples[i,:], RB.N, assemble=True)

        error_max = np.amax(error_estimate)
        sample_index = np.argmax(error_estimate)
        sample_index_set.append(sample_index)
        error_max_set.append(error_max)
        samples_active_set[sample_index] = 0

        # print("error_max", error_max, "RB.tol", RB.tol, "RB.N", RB.N, "RB.Nmax", RB.Nmax)

        if error_max > RB.tol:

            # solve the high fidelity system
            solution = RB.HiFi.solve(samples[sample_index, :])
            if RB.HiFi.misfit is not None:
                solutionAdj = RB.HiFi.solveAdj(samples[sample_index, :], solution)

            # # Gram Schmidt orthogonalization
            # solution_orthogonal = GramSchmidt(basis, solution[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)
            # if RB.HiFi.misfit is not None:
            #     solutionAdj_orthogonal = GramSchmidt(basisAdj, solutionAdj[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)

            solution_orthogonal = solution[RB.HiFi.node_inner]
            if RB.HiFi.misfit is not None:
                solutionAdj_orthogonal = solutionAdj[RB.HiFi.node_inner]

            # expand the basis
            if len(basis) > 0:
                basis = np.hstack((basis, np.array([solution_orthogonal]).T))
            else:
                basis = np.array([solution_orthogonal]).T
            RB.basis = basis
            if RB.HiFi.misfit is not None:
                if len(basisAdj) > 0:
                    basisAdj = np.hstack((basisAdj, np.array([solutionAdj_orthogonal]).T))
                else:
                    basisAdj = np.array([solutionAdj_orthogonal]).T
                RB.basisAdj = basisAdj

            RB.N += 1

            if RB.HiFi.misfit is not None:
                RB.A_rb, RB.f_rb, RB.At_rb, RB.B_V, RB.B_W = SystemProjectionAdjoint(RB, basis, basisAdj)
            else:
                RB.A_rb, RB.f_rb = SystemProjection(RB, basis)

            if RB.HiFi.misfit is not None:
                RB.Wt_fi, RB.Wt_Ai_V = ResidualProjectionAdjoint(RB.HiFi, basis, basisAdj)
            else:
                RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(RB.HiFi, basis)

            # print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)

    RB.error_max = error_max
    RB.error_max_set = error_max_set
    RB.sample_index_set = sample_index_set


#
# def BuildReducedBasisGreedyQoI(RB, HiFi, HiFiAdjoint, samples, tol, Nmax):
#
#     RB.N = 0
#     nSamples = len(samples)
#
#     samples_active_set = np.ones(nSamples)
#     sample_index = 0
#     sample_index_set =[0,]
#     samples_active_set[sample_index] = 0
#     error_max_set = [1,]
#     error_max = tol + 1.
#     basis = []
#     basis_adjoint = []
#
#     while RB.N < Nmax and error_max > tol:
#
#         RB.N += 1
#
#         # solve the high fidelity system
#         solution = HiFi.solve(samples[sample_index,:])
#         solution_adjoint = HiFiAdjoint.solve(samples[sample_index,:])
#
#         # Gram Schmidt orthogonalization
#         solution_orthogonal = GramSchmidt(basis, solution[HiFi.node_inner], HiFi.Xnorm_inner)
#         solution_orthogonal_adjoint = GramSchmidt(basis_adjoint, solution_adjoint[HiFiAdjoint.node_inner], HiFiAdjoint.Xnorm_inner)
#
#         # expand the basis
#         if len(basis) > 0:
#             basis = np.hstack((basis, np.array([solution_orthogonal]).T))
#         else:
#             basis = np.array([solution_orthogonal]).T
#         # print "basis", len(basis), basis.shape
#         RB.basis = basis
#
#         if len(basis_adjoint) > 0:
#             basis_adjoint = np.hstack((basis_adjoint, np.array([solution_orthogonal_adjoint]).T))
#         else:
#             basis_adjoint = np.array([solution_orthogonal_adjoint]).T
#         # print "basis", len(basis), basis.shape
#         RB.basis_adjoint = basis_adjoint
#
#         RB.A_rb, RB.f_rb = SystemProjection(HiFi, basis, RB.projection_type)
#         # print "RB.A_rb, RB.f_rb", len(RB.A_rb), len(RB.f_rb), RB.A_rb[0].shape
#         # RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(HiFi, basis)
#
#         RB.A_rb_adjoint, RB.f_rb_adjoint = SystemProjection(HiFiAdjoint, basis_adjoint, RB.projection_type)
#
#         RB.fi_W, RB.Vt_Ai_W = ResidualProjectionAdjoint(HiFi, basis, basis_adjoint)
#
#         error_estimate = np.zeros(nSamples)
#         for i in range(nSamples):
#             if samples_active_set[i]:
#                 error_estimate[i] = RB.error_estimate_qoi(samples[i,:], RB.N, assemble=True)
#
#         error_max = np.amax(error_estimate)
#         sample_index = np.argmax(error_estimate)
#         sample_index_set.append(sample_index)
#         error_max_set.append(error_max)
#         samples_active_set[sample_index] = 0
#
#         print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)
#
#     RB.error_max_set = error_max_set
#     RB.sample_index_set = sample_index_set
#
#
# def UpdateReducedBasisGreedyQoI(RB, samples, restart=False):
#
#     nSamples = len(samples)
#
#     samples_active_set = np.ones(nSamples)
#     sample_index = 0
#     sample_index_set =[0,]
#     samples_active_set[sample_index] = 0
#     error_max_set = [1,]
#     error_max = RB.tol + 1.
#     if restart:
#         RB.N = 1
#         basis = RB.basis[:,:1]
#         basis_adjoint = RB.basis_adjoint[:,:1]
#     else:
#         basis = RB.basis
#         basis_adjoint = RB.basis_adjoint
#     # basis = RB.basis
#     # basis_adjoint = RB.basis_adjoint
#
#     while RB.N < RB.Nmax and error_max > RB.tol:
#
#
#         error_estimate = np.zeros(nSamples)
#         for i in range(nSamples):
#             if samples_active_set[i]:
#                 error_estimate[i] = RB.error_estimate_qoi(samples[i,:], RB.N, assemble=True)
#
#         error_max = np.amax(error_estimate)
#         sample_index = np.argmax(error_estimate)
#         sample_index_set.append(sample_index)
#         error_max_set.append(error_max)
#         samples_active_set[sample_index] = 0
#
#         if error_max > RB.tol:
#             # solve the high fidelity system
#             solution = RB.HiFi.solve(samples[sample_index,:])
#             solution_adjoint = RB.HiFiAdjoint.solve(samples[sample_index,:])
#
#             # Gram Schmidt orthogonalization
#             solution_orthogonal = GramSchmidt(basis, solution[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)
#             solution_orthogonal_adjoint = GramSchmidt(basis_adjoint, solution_adjoint[RB.HiFiAdjoint.node_inner], RB.HiFiAdjoint.Xnorm_inner)
#
#             RB.N += 1
#
#             # expand the basis
#             if len(basis) > 0:
#                 basis = np.hstack((basis, np.array([solution_orthogonal]).T))
#             else:
#                 basis = np.array([solution_orthogonal]).T
#             # print "basis", len(basis), basis.shape
#             RB.basis = basis
#
#             if len(basis_adjoint) > 0:
#                 basis_adjoint = np.hstack((basis_adjoint, np.array([solution_orthogonal_adjoint]).T))
#             else:
#                 basis_adjoint = np.array([solution_orthogonal_adjoint]).T
#             # print "basis", len(basis), basis.shape
#             RB.basis_adjoint = basis_adjoint
#
#             RB.A_rb, RB.f_rb = SystemProjection(RB.HiFi, basis, RB.projection_type)
#             # print "RB.A_rb, RB.f_rb", len(RB.A_rb), len(RB.f_rb), RB.A_rb[0].shape
#             # RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(HiFi, basis)
#
#             RB.A_rb_adjoint, RB.f_rb_adjoint = SystemProjection(RB.HiFiAdjoint, basis_adjoint, RB.projection_type)
#
#             RB.fi_W, RB.Vt_Ai_W = ResidualProjectionAdjoint(RB.HiFi, basis, basis_adjoint)
#
#             print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)
#
#     RB.error_max_set = error_max_set
#     RB.sample_index_set = sample_index_set
#
#
# def BuildReducedBasisGreedyMisfit(RB, HiFi, HiFiMisfit, samples, tol, Nmax):
#
#     RB.N = 0
#     nSamples = len(samples)
#
#     samples_active_set = np.ones(nSamples)
#     sample_index = 0
#     sample_index_set =[0,]
#     samples_active_set[sample_index] = 0
#     error_max_set = [1,]
#     error_max = tol + 1.
#     basis = []
#     basis_misfit = []
#
#     while RB.N < Nmax and error_max > tol:
#
#         RB.N += 1
#
#         # solve the high fidelity system
#         solution = HiFi.solve(samples[sample_index,:])
#         solution_misfit = HiFiMisfit.solve(samples[sample_index,:], solution)
#
#         # Gram Schmidt orthogonalization
#         solution_orthogonal = GramSchmidt(basis, solution[HiFi.node_inner], HiFi.Xnorm_inner)
#         solution_orthogonal_misfit = GramSchmidt(basis_misfit, solution_misfit[HiFiMisfit.node_inner], HiFiMisfit.Xnorm_inner)
#
#         # expand the basis
#         if len(basis) > 0:
#             basis = np.hstack((basis, np.array([solution_orthogonal]).T))
#         else:
#             basis = np.array([solution_orthogonal]).T
#         # print "basis", len(basis), basis.shape
#         RB.basis = basis
#
#         if len(basis_misfit) > 0:
#             basis_misfit = np.hstack((basis_misfit, np.array([solution_orthogonal_misfit]).T))
#         else:
#             basis_misfit = np.array([solution_orthogonal_misfit]).T
#         # print "basis", len(basis), basis.shape
#         RB.basis_misfit = basis_misfit
#
#         RB.A_rb, RB.f_rb = SystemProjection(HiFi, basis, RB.projection_type)
#         # print "RB.A_rb, RB.f_rb", len(RB.A_rb), len(RB.f_rb), RB.A_rb[0].shape
#         # RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(HiFi, basis)
#
#         RB.A_rb_misfit, RB.f_rb_misfit = SystemProjection(HiFiMisfit, basis_misfit, RB.projection_type)
#
#         # gradient of misfit term, B_W = (B * basis_misfit), B_V = (B * basis) matrix of (s x N)
#         B_V = np.zeros([len(HiFiMisfit.misfit.d.get_local()), RB.N])
#         B_W = np.zeros([len(HiFiMisfit.misfit.d.get_local()), RB.N])
#         for n in range(RB.N):
#             HiFiMisfit.uhelp_array[HiFiMisfit.node_inner] = basis[:,n]
#             HiFiMisfit.uhelp.set_local(HiFiMisfit.uhelp_array)
#             HiFiMisfit.misfit.B.mult(HiFiMisfit.uhelp, HiFiMisfit.misfit.help)
#             B_V[:, n] = HiFiMisfit.misfit.help.get_local()
#
#             HiFiMisfit.uhelp_array[HiFiMisfit.node_inner] = basis_misfit[:,n]
#             HiFiMisfit.uhelp.set_local(HiFiMisfit.uhelp_array)
#             HiFiMisfit.misfit.B.mult(HiFiMisfit.uhelp, HiFiMisfit.misfit.help)
#             B_W[:, n] = HiFiMisfit.misfit.help.get_local()
#         RB.B_V, RB.B_W = B_V, B_W
#
#         RB.fi_W, RB.Vt_Ai_W = ResidualProjectionAdjoint(HiFi, basis, basis_misfit)
#
#         error_estimate = np.zeros(nSamples)
#         for i in range(nSamples):
#             if samples_active_set[i]:
#                 error_estimate[i] = np.abs(RB.error_estimate_misfit(samples[i,:], RB.N, assemble=True))
#
#         error_max = np.amax(error_estimate)
#         sample_index = np.argmax(error_estimate)
#         sample_index_set.append(sample_index)
#         error_max_set.append(error_max)
#         samples_active_set[sample_index] = 0
#
#         print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)
#
#     RB.error_max_set = error_max_set
#     RB.sample_index_set = sample_index_set
#
#
# def UpdateReducedBasisGreedyMisfit(RB, samples, restart=False):
#
#     nSamples = len(samples)
#
#     samples_active_set = np.ones(nSamples)
#     sample_index = 0
#     sample_index_set =[0,]
#     samples_active_set[sample_index] = 0
#     error_max_set = [1,]
#     error_max = RB.tol + 1.
#     if restart:
#         RB.N = 1
#         basis = RB.basis[:,:1]
#         basis_misfit = RB.basis_misfit[:,:1]
#     else:
#         basis = RB.basis
#         basis_misfit = RB.basis_misfit
#     HiFiMisfit = RB.HiFiMisfit
#
#     while RB.N < RB.Nmax and error_max > RB.tol:
#
#         error_estimate = np.zeros(nSamples)
#         for i in range(nSamples):
#             if samples_active_set[i]:
#                 error_estimate[i] = np.abs(RB.error_estimate_misfit(samples[i,:], RB.N, assemble=True))
#
#         error_max = np.amax(error_estimate)
#         sample_index = np.argmax(error_estimate)
#         sample_index_set.append(sample_index)
#         error_max_set.append(error_max)
#         samples_active_set[sample_index] = 0
#
#         if error_max > RB.tol:
#             # solve the high fidelity system
#             solution = RB.HiFi.solve(samples[sample_index,:])
#             solution_misfit = RB.HiFiMisfit.solve(samples[sample_index,:], solution)
#
#             # rb_solution = RB.reconstruct(samples[sample_index,:])
#             # rb_solution_misfit = RB.reconstructMisfit(samples[sample_index,:])
#             # print("error_fwd = ", np.dot(np.dot(RB.HiFi.Xnorm, solution-rb_solution), solution-rb_solution)/np.dot(np.dot(RB.HiFi.Xnorm, solution), solution),
#             #       "error_adj = ", np.dot(np.dot(RB.HiFi.Xnorm, solution_misfit-rb_solution_misfit), solution_misfit-rb_solution_misfit)/np.dot(np.dot(RB.HiFi.Xnorm, solution_misfit), solution_misfit))
#
#             # Gram Schmidt orthogonalization
#             solution_orthogonal = GramSchmidt(basis, solution[RB.HiFi.node_inner], RB.HiFi.Xnorm_inner)
#             solution_orthogonal_misfit = GramSchmidt(basis_misfit, solution_misfit[RB.HiFiMisfit.node_inner], RB.HiFiMisfit.Xnorm_inner)
#
#             RB.N += 1
#
#             # expand the basis
#             if len(basis) > 0:
#                 basis = np.hstack((basis, np.array([solution_orthogonal]).T))
#             else:
#                 basis = np.array([solution_orthogonal]).T
#             # print "basis", len(basis), basis.shape
#             RB.basis = basis
#
#             if len(basis_misfit) > 0:
#                 basis_misfit = np.hstack((basis_misfit, np.array([solution_orthogonal_misfit]).T))
#             else:
#                 basis_misfit = np.array([solution_orthogonal_misfit]).T
#             # print "basis", len(basis), basis.shape
#             RB.basis_misfit = basis_misfit
#
#             RB.A_rb, RB.f_rb = SystemProjection(RB.HiFi, basis, RB.projection_type)
#             # print "RB.A_rb, RB.f_rb", len(RB.A_rb), len(RB.f_rb), RB.A_rb[0].shape
#             # RB.fj_Xinv_fi, RB.fj_Xinv_Ai_V, RB.Vt_Aj_Xinv_Ai_V = ResidualProjection(HiFi, basis)
#
#             RB.A_rb_misfit, RB.f_rb_misfit = SystemProjection(RB.HiFiMisfit, basis_misfit, RB.projection_type)
#
#             # gradient of misfit term, B_W = (B * basis_misfit), B_V = (B * basis) matrix of (s x N)
#             B_V = np.zeros([len(HiFiMisfit.misfit.d.get_local()), RB.N])
#             B_W = np.zeros([len(HiFiMisfit.misfit.d.get_local()), RB.N])
#             for n in range(RB.N):
#                 HiFiMisfit.uhelp_array[HiFiMisfit.node_inner] = basis[:,n]
#                 HiFiMisfit.uhelp.set_local(HiFiMisfit.uhelp_array)
#                 HiFiMisfit.misfit.B.mult(HiFiMisfit.uhelp, HiFiMisfit.misfit.help)
#                 B_V[:, n] = HiFiMisfit.misfit.help.get_local()
#
#                 HiFiMisfit.uhelp_array[HiFiMisfit.node_inner] = basis_misfit[:,n]
#                 HiFiMisfit.uhelp.set_local(HiFiMisfit.uhelp_array)
#                 HiFiMisfit.misfit.B.mult(HiFiMisfit.uhelp, HiFiMisfit.misfit.help)
#                 B_W[:, n] = HiFiMisfit.misfit.help.get_local()
#             RB.B_V, RB.B_W = B_V, B_W
#
#             RB.fi_W, RB.Vt_Ai_W = ResidualProjectionAdjoint(RB.HiFi, basis, basis_misfit)
#
#             print("# basis = ", RB.N, "error_max = ", error_max, "sample_index", sample_index)
#
#     RB.error_max_set = error_max_set
#     RB.sample_index_set = sample_index_set