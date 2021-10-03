import numpy as np

class BasisConstructionPOD:

    def __init__(self, snapshots, method, tol=0, Nmax=0):

        self.method = method
        self.tol = tol
        self.Nmax = Nmax
        self.N = 0
        self.snapshots = snapshots

        if self.method == 'SVD':
            self.sigma, self.basis = self.SingularValueDecomposition()
        elif self.method == 'EVD':
            self.sigma, self.basis = self.EigenValueDecomposition()
        else:
            raise NameError('choose method as SVD or EVD')

    def SingularValueDecomposition(self):
        U, sigma, V = np.linalg.svd(self.snapshots, full_matrices=False)
        if self.tol != 0:
            sigma2 = np.cumsum(sigma**2)
            tol = 1.- sigma2/np.sum(sigma**2)
            self.N = next(i+1 for i in range(len(sigma)) if self.tol > tol[i])
            if self.Nmax > 0:
                self.N = np.min([self.N, self.Nmax])
        elif self.Nmax != 0:
            self.N = np.min([self.Nmax, U.shape[1]])
        else:
            self.N = U.shape[1]

        basis = U[:,:self.N]
        sigma = sigma[:self.N]

        return sigma, basis

    def EigenValueDecomposition(self):
        basis = []
        sigma = []
        return sigma, basis
