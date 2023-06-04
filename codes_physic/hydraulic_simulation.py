#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def hydraulic_simulation(C, wdn, d, h0):
    nt = d.shape[1]
    np = wdn['np']
    nn = wdn['nn']
    nulldata = wdn['nulldata']

    # define parameters for the hydraulic solver
    max_iter = 100
    kappa = 1e7
    tol_err = 1e-9

    h = np.zeros((nn, nt))
    q = np.zeros((np, nt))

   

 # Perform extended period simulation
    for t in range(nt):
        q_0 = 0.3 * np.ones((np, 1))
        h_0 = 130 * np.ones((nn, 1))

        # This is a vector required by the null space hydraulic solver
        nulldata['x'] = nulldata['A12'] @ (nulldata['Pr'].T @ (np.linalg.solve(nulldata['L_A12'], np.linalg.solve(nulldata['L_A12'], nulldata['Pr'] @ d[:, t]))))

        # Define the resistance coefficient r
        ResCoeff = 10.670 * wdn['L'] / ((C**1.852) * (wdn['D']**4.871))
        if 'valves' in wdn and len(wdn['valves']) > 0:
            # Define resistance coefficient specific to valves using minor loss coefficient
            valves = wdn['valves']
            ResCoeff[valves] = (8 / (np.pi**2 * 9.81)) * (wdn['D'][valves]**-4) * C[valves]

        qt, ht, err, iter_, CONDS, ERRORS = solveHW_Nullspace(wdn['A12'], wdn['A12'].T, wdn['A10'], h0[:, t], q_0, h_0, d[:, t], np, nulldata, max_iter, kappa, tol_err)

        if ERRORS > tol_err:
            check = 1

        h[:, t] = ht
        q[:, t] = qt

    return q, h

