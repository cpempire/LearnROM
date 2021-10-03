import scipy.linalg as la
import scipy.sparse.linalg as spla
import numpy as np

# import dolfin as dl
#
# nx=64
# ny=64
# mesh = dl.UnitSquareMesh(nx, ny)
#
# Vh = dl.FunctionSpace(mesh,'CG',1)
# dofs = Vh.dofmap().dofs().size
#
# u = dl.TrialFunction(Vh)
# v = dl.TestFunction(Vh)
#
# M = dl.assemble(u*v*dl.dx)
#
# M = M.array()
#
# import time
# t0 = time.time()
# Msr = la.sqrtm(M)
# print "square root time = ", time.time() - t0
# # Msr = np.sqrt(M)
#
# A = np.random.rand(dofs,dofs)
# B = np.matmul(Msr, A)
#
# # S, Sigma, Q = np.linalg.svd(B)
# S, Sigma, Q = spla.svds(B, k=20, which='LM')
# U = []
# for n in range(20):
#     v = 1./Sigma[n]*np.dot(A, Q[n,:])
#     print "v^TCv = ", np.dot(v, np.dot(M, v))
#


R = np.random.rand(1000,1000)
C = np.matmul(R.T, R)

R = np.linalg.cholesky(C)

D = np.matmul(R, R.T) - C
print D

A = np.random.rand(1000,200)
B = np.matmul(R, A)

# B = sp.sparse.csc_matrix(B)
# S, Sigma, Q = sp.sparse.linalg.svds(B, k=20, which='LM')

S, Sigma, Q = np.linalg.svd(B)

U = []
for n in range(100):
    v = 1./Sigma[n]*np.dot(A, Q[n,:])
    print "v^TCv = ", np.dot(v, np.dot(C, v))
    v = np.linalg.solve(R.T, S[:,n])
    print "v^TCv = ", np.dot(v, np.dot(C, v))
