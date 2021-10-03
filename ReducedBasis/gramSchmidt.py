import numpy as np

def GramSchmidt(basis, b, Xnorm=None):

    if Xnorm is None:
        Xnorm = np.eye(len(b),dtype=float)

    if len(basis) > 0:
        coeff = np.dot(basis.T, Xnorm.dot(b))
        Rb = b - np.dot(basis, coeff)

        # loss of orthogonality
        if np.linalg.norm(Rb) < 0.7*np.linalg.norm(b):
            coeff = np.dot(basis.T, Xnorm.dot(Rb))
            Rb = Rb - np.dot(basis, coeff)

    else:
        Rb = b

    if np.linalg.norm(Rb) > 2e-16:
        Rb /= np.sqrt(np.dot(Rb, Xnorm.dot(Rb)))

    return Rb
