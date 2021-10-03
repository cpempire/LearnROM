from .basisConstructionPOD import BasisConstructionPOD
from .systemProjection import SystemProjection

def BuildReducedBasisPOD(RB, HiFi, samples, tol, Nmax):

    # generate snapshots
    snapshots = HiFi.snapshots(samples)

    # compute POD basis functions
    snapshots = snapshots[HiFi.node_inner, :]
    basis, sigma, N = BasisConstructionPOD(snapshots,'SVD',tol=tol,Nmax=Nmax)

    RB.N = N
    RB.basis = basis
    RB.sigma = sigma

    if HiFi.problem_type is 'affine':
        # project the high fidelity system to a reduced basis system
        A_rb, f_rb = SystemProjection(HiFi, basis, RB.projection_type)

        RB.A_rb = A_rb
        RB.f_rb = f_rb

    # return basis, sigma, N, A_rb, f_rb


# class BuildReducedBasisPOD:
#
#     def __init__(RB, HiFi, samples, method, tol, Nmax):
#
#         # parameters
#         # RB.num_para  = HiFi.num_para
#         # RB.para_mean = HiFi.para_mean
#         # RB.para_min  = HiFi.para_min
#         # RB.para_max  = HiFi.para_max
#         if hasattr(HiFi, 'problem_type'):
#             RB.problem_type = HiFi.problem_type
#             RB.HiFi = HiFi
#
#         RB.Qa        = HiFi.Qa
#         RB.Qf        = HiFi.Qf
#
#         # boundary/inner index and data
#         RB.node_inner = HiFi.node_inner
#         RB.node_Dirichlet = HiFi.node_Dirichlet
#         RB.dofs = HiFi.dofs
#         RB.u_Dirichlet = HiFi.u_Dirichlet
#
#         # coefficient evaluation
#         RB.coeffEvaluation = HiFi.coeffEvaluation
#
#         # compute snapshots at training samples
#         print "generate snapshots"
#         t = time.clock()
#         snapshots = HiFi.snapshots(samples)
#         print "generate snapshots time = ", time.clock()-t
#
#         # compute POD basis functions
#         snapshots_inner = snapshots[RB.node_inner, :]
#         basisPOD = BasisConstructionPOD(snapshots_inner,'SVD',tol=tol,Nmax=Nmax)
#         RB.basis = basisPOD.basis
#         RB.sigma = basisPOD.sigma
#         RB.N = basisPOD.N
#
#         # project the system
#         RB.method = method
#         RB.A_rb, RB.f_rb = SystemProjection(HiFi, RB.basis, RB.method)
#
#     def solve(RB, sample, N=0, assemble=True):
#         coeff_a, coeff_f = RB.coeffEvaluation(sample)
#         if hasattr(RB, 'problem_type'):
#             if RB.problem_type == 'nonaffine' and assemble:
#                 # print "assemble", assemble
#                 # assemble HiFi matrix and vector and project them to RB matrix and vector
#                 RB.HiFi.assemble(sample)
#                 RB.A_rb, RB.f_rb = SystemProjection(RB.HiFi, RB.basis, RB.method)
#
#         if N == 0:
#             N = RB.N
#
#         if RB.method == 'Galerkin':
#             A = coeff_a[0]*RB.A_rb[0][:N, :N]
#             for i in range(1, RB.Qa):
#                 A += coeff_a[i]*RB.A_rb[i][:N, :N]
#
#             f = coeff_f[0]*RB.f_rb[0][:N]
#             for i in range(1, RB.Qf):
#                 f += coeff_f[i]*RB.f_rb[i][:N]
#
#             if RB.HiFi.bc is not None and np.linalg.norm(RB.HiFi.u_Dirichlet,np.inf) > 2e-16:
#                 for i in range(RB.Qa):
#                     f -= coeff_a[i]*RB.f_rb[RB.Qf+i][:N]
#
#         elif RB.method == 'PetrovGalerkin':
#
#             A = 0.*RB.A_rb[0][:N, :N]
#             f = 0.*RB.f_rb[0][:N]
#             for i in range(RB.Qa):
#                 for j in range(RB.Qa):
#                     A += coeff_a[i]*coeff_a[j]*RB.A_rb[i*RB.Qa+j][:N,:N]
#                 for j in range(RB.Qf):
#                     f += coeff_a[i]*coeff_f[j]*RB.f_rb[i*RB.Qa+j][:N]
#
#                 if RB.HiFi.bc is not None and np.linalg.norm(RB.HiFi.u_Dirichlet,np.inf) > 2e-16:
#                     for j in range(RB.Qa):
#                         f -= coeff_a[j]*RB.f_rb[RB.Qf+j][:N]
#
#         else:
#             raise NameError('choose between Galerkin and PetrovGalerkin')
#
#         uN = np.linalg.solve(A, f)
#
#         return uN
#
#     def reconstruct(RB, sample, N, assemble=True):
#         uN = RB.solve(sample, N, assemble)
#
#         u_rb = uN[0]*RB.basis[:,0]
#         for i in range(1, N):
#             u_rb += uN[i]*RB.basis[:,i]
#
#         # # print "dofs",RB.dofs
#         solution = np.zeros(RB.dofs)
#         solution[RB.node_inner] = u_rb
#         solution[RB.node_Dirichlet] = RB.u_Dirichlet
#
#         # solution = u_rb
#         # if N >= 90:
#         # u_fun = dl.Function(RB.HiFi.Vh[0])
#         # u_fun.vector().set_local(solution)
#         # dl.plot(u_fun)
#         # dl.interactive()
#
#         return solution
#
#     def snapshots(RB, samples, N=0, assemble=True):
#         if N == 0:
#             N = RB.N
#
#         num_samples = samples.shape[0]
#
#         U = []
#         for i in range(num_samples):
#             sample = samples[i, :]
#             u = RB.reconstruct(sample, N, assemble)
#             U.append(u)
#         U = np.array(U).T
#         return U