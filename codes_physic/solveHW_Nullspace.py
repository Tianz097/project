import numpy as np
from scipy.sparse import spdiags, csc_matrix

def solveHW_Nullspace(A12, A21, A10, h0, qk, hk, d, np, nulldata, data):
    ResCoeff = data['ResCoeff']
    x = nulldata['x']
    Z = nulldata['Z']
    P_r = nulldata['Pr']
    L_A12 = nulldata['L_A12']

    CONDS = 1
    ERRORS = 1

    max_iter = data['max_iter']
    tol = data['tol_err']
    kappa = data['kappa']

    n_exp = 1.852 * np.ones((np, 1))
    if 'valves' in data and len(data['valves']) > 0:
        n_exp[data['valves']] = 2

    A11 = ResCoeff * np.abs(qk) ** (n_exp - 1)
    err1 = np.linalg.norm([A11 * qk + A12 * hk + A10 * h0, A21 * qk - d], np.inf)

    nc = Z.shape[1]
    nnz_ZZ = np.count_nonzero(Z.T @ spdiags(np.ones(np), 0, np, np) @ Z)
    Fdiag_old = np.zeros((np, 1))
    X = np.zeros((nc, nc))
    updates = np.arange(1, np + 1)
    n_upd = len(updates)

    for kk in range(1, max_iter + 1):
        Fdiag = n_exp * ResCoeff * np.abs(qk) ** (n_exp - 1)

        sigma_max = np.max(Fdiag)
        t_k = np.maximum((sigma_max / kappa) - Fdiag, 0)
        Fdiag = Fdiag + t_k

        X = Z.T @ spdiags(Fdiag, 0, np, np) @ Z
        b = Z.T @ ((Fdiag - A11) * qk - A10 * h0 - Fdiag * x)
        v = csc_matrix(X).solve(b)

        q = x + Z @ v
        b = A12.T @ ((Fdiag - A11) * qk - A10 * h0 - Fdiag * q)
        y = np.linalg.solve(L_A12, P_r @ b)
        h = P_r.T @ np.linalg.solve(L_A12.T, y)

        A11 = ResCoeff * (np.abs(q) ** (n_exp - 1))

        err1 = np.linalg.norm([A11 * q + A12 * h + A10 * h0, A21 * q - d], np.inf)
        ERRORS = err1

        if err1 < tol:
            break
        else:
            qk = q

    return q, h, err1, kk, CONDS, ERRORS
