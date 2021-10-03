import numpy as np

def ResidualProjection(HiFi, basis):

    L = np.linalg.cholesky(HiFi.Xnorm_inner)

    fj_Xinv_fi = ()
    for i in range(HiFi.Qf_rb):
        Xinv_fi = np.linalg.solve(L, HiFi.f_hifi_inner[i])
        Xinv_fi = np.linalg.solve(L.T, Xinv_fi)
        for j in range(HiFi.Qf_rb):
            fj_Xinv_fi += (np.dot(HiFi.f_hifi_inner[j], Xinv_fi),)

    fj_Xinv_Ai_V = ()
    Vt_Aj_Xinv_Ai_V = ()
    for i in range(HiFi.Qa_rb):
        Xinv_Ai_V = np.linalg.solve(L, HiFi.A_hifi_inner[i].dot(basis))
        Xinv_Ai_V = np.linalg.solve(L.T, Xinv_Ai_V)
        for j in range(HiFi.Qf_rb):
            fj_Xinv_Ai_V += (np.dot(HiFi.f_hifi_inner[j], Xinv_Ai_V), )

        for j in range(HiFi.Qa_rb):
            Vt_Aj_Xinv_Ai_V += (np.dot(HiFi.A_hifi_inner[j].dot(basis).T, Xinv_Ai_V), )

    return fj_Xinv_fi, fj_Xinv_Ai_V, Vt_Aj_Xinv_Ai_V


def ResidualProjectionAdjoint(HiFi, basis, basis_adjoint):
    Wt_fi = ()
    for i in range(HiFi.Qf_rb):
        Wt_fi += (np.dot(basis_adjoint.T, HiFi.f_hifi_inner[i]), )

    Wt_Ai_V = ()
    for i in range(HiFi.Qa_rb):
        Wt_Ai_V += (np.dot(basis_adjoint.T, HiFi.A_hifi_inner[i].dot(basis)),)

    return Wt_fi, Wt_Ai_V