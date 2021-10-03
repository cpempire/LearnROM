import numpy as np

def BasisConstructionPOD(snapshots, method, tol=0, Nmax=0):
    "basis construction by POD, either SVD or EVD"
    if method is 'SVD':
        U, sigma, V = np.linalg.svd(snapshots, full_matrices=False)
        if tol != 0:
            sigma2 = np.cumsum(sigma**2)
            sigma2_res = 1.- sigma2/np.sum(sigma**2)
            N = next(i+1 for i in range(len(sigma)) if tol > sigma2_res[i])
            if Nmax > 0:
                N = np.min([N, Nmax])
        elif Nmax != 0:
            N = np.min([Nmax, U.shape[1]])
        else:
            N = U.shape[1]

        basis = U[:,:N]
        sigma = sigma[:N]

    elif method is 'EVD':
        basis = []
        sigma = []
        N = []
    else:
        raise NameError('method is SVD or EVD')

    return basis, sigma, N