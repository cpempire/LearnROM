import numpy as np

class SystemProjection:
    """Project the system by Galerkin or PetrovGalerkin method"""
    def __init__(self, HiFi, basis, method):
        """
        :param HiFi: HiFi class
        :param basis: basis functions
        :param method: Galerkin, PetrovGalerkin
        :return: A_rb, f_rb
        """
        self.HiFi = HiFi
        self.basis = basis
        self.method = method

        if method == 'Galerkin':
            self.A_rb, self.f_rb = self.GalerkinProjection()
        elif method == 'PetrovGalerkin': # Least Squares PG
            self.A_rb, self.f_rb = self.PetrovGalerkinProjection()

    def GalerkinProjection(self):
        A_rb = ()
        for i in range(self.HiFi.Qa):
            A_hifi_i = self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)]
            # print "A_hifi_i, basis", A_hifi_i.shape, self.basis.shape
            Atemp = np.dot(self.basis.T, A_hifi_i.dot(self.basis))
            A_rb += (Atemp, )

        f_rb = ()
        for i in range(self.HiFi.Qf):
            f_hifi_i = self.HiFi.f_hifi[i].array()[self.HiFi.node_inner]
            ftemp = np.dot(self.basis.T, f_hifi_i)
            f_rb += (ftemp, )

        return A_rb, f_rb

    def PetrovGalerkinProjection(self):
        A_rb = ()
        f_rb = ()
        for i in range(self.HiFi.Qa):
            for j in range(self.HiFi.Qa):
                A_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T,
                                   self.HiFi.A_hifi[j].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)])
                Atemp = np.dot(self.basis.T, A_hifi_ij.dot(self.basis))
                A_rb += (Atemp, )

            for j in range(self.HiFi.Qf):
                f_hifi_ij = np.dot(self.HiFi.A_hifi[i].array()[np.ix_(self.HiFi.node_inner, self.HiFi.node_inner)].T,
                                   self.HiFi.f_hifi[j].array()[self.HiFi.node_inner])
                ftemp = np.dot(self.basis.T, f_hifi_ij)
                f_rb += (ftemp, )

        return A_rb, f_rb
