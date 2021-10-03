import numpy as np

class EmpiricalInterpolation:

    def __init__(self, train_samples, tol=1e-8, Nmax=0, norm=np.inf):
        self.nodes = []
        self.bases = []
        self.N = 0

        self.error = []
        self.index = []

        self.construction(train_samples, tol=tol, Nmax=Nmax, norm=norm)

    def construction(self, train_samples, tol=1e-8, Nmax=0, norm=None):

        (Nh, Ns) = train_samples.shape
        error = np.zeros(Ns)

        if Nmax == 0:
            Nmax = Ns

        for i in range(Ns):
            # error[i] = np.max(np.abs(train_samples[:,i]))
            error[i] = np.linalg.norm(train_samples[:,i],norm)

        index_max = np.argmax(error)
        index_set = np.setdiff1d(range(Ns),index_max)
        error[index_max] = 0
        index_max_set = []
        index_max_set += [index_max,]
        res = train_samples[:,index_max]

        N = 0
        nodes = []
        bases = np.zeros([Nh, Ns])
        error_EIM = [2*tol]

        while error_EIM[N] > tol and N < Nmax:

            N += 1

            node_max = np.argmax(np.abs(res))
            nodes += [node_max,]
            # bases[:,N-1] = res/np.abs(res[node_max])
            bases[:,N-1] = res/np.linalg.norm(res,norm)
            A = bases[nodes,:N]

            for i in index_set:
                fun_i = train_samples[:,i]
                coeff = np.linalg.solve(A, fun_i[nodes])
                res = fun_i - np.dot(bases[:,:N], coeff)
                # error[i] = np.max(np.abs(res)) #/np.max(np.abs(fun_i)))
                error[i] = np.linalg.norm(res,norm)

            index_max = np.argmax(error)
            error_EIM += [error[index_max],]
            index_set = np.setdiff1d(index_set,index_max)
            error[index_max] = 0
            index_max_set += [index_max,]

            fun_max = train_samples[:,index_max]
            coeff = np.linalg.solve(A, fun_max[nodes])
            res = fun_max - np.dot(bases[:,:N], coeff)

        # from IPython import embed
        # embed()
        self.bases = bases[:,:N]
        self.nodes = nodes
        self.N = N

        self.error = error_EIM
        self.index = index_max_set


    def error_test(self,test_funcs,norm=np.inf,plot=False):
        (Nh, Ns) = test_funcs.shape
        errors = np.zeros([self.N, Ns])


        for i in range(self.N):
            A = self.bases[:,:i+1]
            for j in range(Ns):
                b = test_funcs[self.nodes[:i+1],j]
                coeff = np.linalg.solve(A[self.nodes[:i+1],:], b)
                res = test_funcs[:,j] - np.dot(A, coeff)
                errors[i,j] = np.linalg.norm(res,norm)
                # if norm == 'Linfty':
                #     errors[i,j] = np.max(np.abs(res))
                # elif norm == 'L2':
                #     errors[i,j] = np.sqrt(np.dot(res,res))


        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            x = range(1, self.N+1)
            error_max = np.max(errors, axis=1)
            error_mean= np.mean(errors, axis=1)
            plt.semilogy(x, error_max,'*-',label='max error')
            plt.semilogy(x, error_mean,'*-',label='mean error')
            plt.xlabel('N')
            plt.ylabel('error')
            plt.legend()
            plt.show()

        return errors

class DiscreteEmpiricalInterpolation:

    def __init__(self, train_samples, tol=1e-14, Nmax=0, norm=np.inf):

        self.nodes = []
        self.bases = []
        self.N = 0

        self.error = []
        self.sigma = []

        self.construction(train_samples, tol=tol, Nmax=Nmax)


    def construction(self, train_samples, tol=1e-14, Nmax=0):


        (Nh, Ns) = train_samples.shape

        if Nmax == 0:
            Nmax = Ns

        # for i in range(Ns):
        #     train_samples[:,i] = train_samples[:,i]/np.linalg.norm(train_samples[:,i])

        U, sigma, V = np.linalg.svd(train_samples, full_matrices=False)
        if tol != 0:
            sigma2 = np.cumsum(sigma**2)
            error = 1.- sigma2/np.sum(sigma**2)
            Nmax = next(i+1 for i in range(len(sigma)) if tol > error[i])

        # Nmax = next(i+1 for i in range(Nmax) if np.abs(sigma[i])>1e-15)

        # plt.figure()
        # plt.semilogy(np.abs(sigma),'*-')
        # plt.show()

        N = 1
        nodes = []
        bases = U[:,:Nmax]
        res = bases[:,0]
        node_max = np.argmax(np.abs(res))
        nodes += [node_max,]
        error_EIM = [np.linalg.norm(res)]

        while N < Nmax:

            A = bases[nodes,:N]

            coeff = np.linalg.solve(A,bases[nodes,N])
            res = bases[:,N]- np.dot(bases[:,:N], coeff)
            node_max = np.argmax(np.abs(res))
            nodes += [node_max,]
            error_EIM += [np.linalg.norm(res),]

            N += 1

        self.nodes = nodes
        self.bases = bases
        self.N = N
        self.error = error_EIM
        self.sigma = sigma

    def error_test(self,test_funcs,norm=np.inf,plot=False):
        (Nh, Ns) = test_funcs.shape
        errors = np.zeros([self.N, Ns])

        for i in range(self.N):
            A = self.bases[:,:i+1]
            for j in range(Ns):
                b = test_funcs[self.nodes[:i+1],j]
                coeff = np.linalg.solve(A[self.nodes[:i+1],:], b)
                res = test_funcs[:,j] - np.dot(A, coeff)
                errors[i,j] = np.linalg.norm(res,norm)
                # if norm == 'Linfty':
                #     errors[i,j] = np.max(np.abs(res))
                # elif norm == 'L2':
                #     errors[i,j] = np.sqrt(np.dot(res,res))

        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            x = range(1, self.N+1)
            error_max = np.max(errors, axis=1)
            error_mean= np.mean(errors, axis=1)
            plt.semilogy(x, error_max,'*-',label='max error')
            plt.semilogy(x, error_mean,'*-',label='mean error')
            plt.xlabel('N')
            plt.ylabel('error')
            plt.legend()
            plt.show()

        return errors


# def DiscreteEmpiricalInterpolation(train_samples, tol=1e-14, Nmax=0):
#
#     (Nh, Ns) = train_samples.shape
#
#     if Nmax == 0:
#         Nmax = Ns
#
#     # for i in range(Ns):
#     #     train_samples[:,i] = train_samples[:,i]/np.linalg.norm(train_samples[:,i])
#
#     U, sigma, V = np.linalg.svd(train_samples, full_matrices=False)
#     if tol != 0:
#         sigma2 = np.cumsum(sigma**2)
#         error = 1.- sigma2/np.sum(sigma**2)
#         Nmax = next(i+1 for i in range(len(sigma)) if tol > error[i])
#
#     Nmax = next(i+1 for i in range(len(sigma)) if np.abs(sigma[i])<1e-15)
#
#     plt.figure()
#     plt.semilogy(np.abs(sigma),'*-')
#     plt.show()
#
#     # print "U", U.shape
#     #
#     # plt.figure()
#     # plt.semilogy(np.abs(sigma),'*-')
#     # plt.show()
#
#     # for i in range(Ns):
#     #     error[i] = np.max(np.abs(train_samples[:,i]))
#     #
#     # index_max = np.argmax(error)
#     # index_set = np.setdiff1d(range(Ns),index_max)
#     # error = np.zeros(Ns)
#     # error[index_max] = 0
#     # index_max_set = []
#     # index_max_set += [index_max,]
#     # res = train_samples[:,index_max]
#
#     N = 0
#     nodes = []
#     bases = U[:,:Nmax]
#     res = bases[:,0]
#     node_max = np.argmax(np.abs(res))
#     nodes += [node_max,]
#     error_EIM = [np.linalg.norm(res)]
#
#
#     for N in range(1,Nmax):
#
#         print "N/Nmax", N, Nmax
#
#         A = bases[nodes,:N]
#
#         coeff = np.linalg.solve(A,bases[nodes,N])
#         res = bases[:,N]- np.dot(bases[:,:N], coeff)
#         node_max = np.argmax(np.abs(res))
#         nodes += [node_max,]
#         error_EIM += [np.linalg.norm(res),]
#         # for i in index_set:
#         #     fun_i = train_samples[:,i]
#         #     coeff = np.linalg.solve(A, fun_i[nodes])
#         #     res = fun_i - np.dot(bases[:,:N], coeff)
#         #     error[i] = np.max(np.abs(res)) #/np.max(np.abs(fun_i)))
#         #
#         # index_max = np.argmax(error)
#         # error_EIM += [error[index_max],]
#         # index_set = np.setdiff1d(index_set,index_max)
#         # error[index_max] = 0
#         # index_max_set += [index_max,]
#         #
#         # fun_max = train_samples[:,index_max]
#         # coeff = np.linalg.solve(A, fun_max[nodes])
#         # res = fun_max - np.dot(bases[:,:N], coeff)
#
#     # from IPython import embed
#     # embed()
#     DEIM = {'nodes':nodes,'bases':bases,'N':Nmax,'error_EIM':error_EIM,'sigma':sigma}
#     return DEIM
#
#
#
#
#
#
#
# def EmpiricalInterpolation(train_samples, tol=1e-8, Nmax=0):
#
#     (Nh, Ns) = train_samples.shape
#     error = np.zeros(Ns)
#
#     if Nmax == 0:
#         Nmax = Ns
#
#     for i in range(Ns):
#         error[i] = np.max(np.abs(train_samples[:,i]))
#
#     index_max = np.argmax(error)
#     index_set = np.setdiff1d(range(Ns),index_max)
#     error[index_max] = 0
#     index_max_set = []
#     index_max_set += [index_max,]
#     res = train_samples[:,index_max]
#
#     N = 0
#     nodes = []
#     bases = np.zeros([Nh, Ns])
#     error_EIM = [2*tol]
#
#     while error_EIM[N] > tol and N < Nmax:
#
#         N += 1
#
#         node_max = np.argmax(np.abs(res))
#         nodes += [node_max,]
#         bases[:,N-1] = res/np.abs(res[node_max])
#         A = bases[nodes,:N]
#
#         for i in index_set:
#             fun_i = train_samples[:,i]
#             coeff = np.linalg.solve(A, fun_i[nodes])
#             res = fun_i - np.dot(bases[:,:N], coeff)
#             error[i] = np.max(np.abs(res)) #/np.max(np.abs(fun_i)))
#
#         index_max = np.argmax(error)
#         error_EIM += [error[index_max],]
#         index_set = np.setdiff1d(index_set,index_max)
#         error[index_max] = 0
#         index_max_set += [index_max,]
#
#         fun_max = train_samples[:,index_max]
#         coeff = np.linalg.solve(A, fun_max[nodes])
#         res = fun_max - np.dot(bases[:,:N], coeff)
#
#     # from IPython import embed
#     # embed()
#     bases = bases[:,:N]
#     EIM = {'nodes':nodes,'bases':bases,'N':N,'error_EIM':error_EIM,'index_set':index_max_set}
#     return EIM
#
