#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def linear_approx_calibration(wdn, q, theta):
    ResCoeff = 10.670 * wdn['L'] / ((theta ** wdn['n_exp']) * (wdn['D'] ** 4.871))

    if 'valves' in wdn and len(wdn['valves']) > 0:
        # 根据小损失系数定义与阀门相关的粗糙度系数
        wdn['n_exp'][wdn['valves']] = 2
        ResCoeff[wdn['valves']] = (8 / (np.pi ** 2 * 9.81)) * (wdn['D'][wdn['valves']] ** -4) * theta[wdn['valves']]

    a11k = np.tile(ResCoeff, (1, q.shape[1])) * np.abs(q) ** (wdn['n_exp'] - 1)

    b1k = ResCoeff
    if 'valves' in wdn and len(wdn['valves']) > 0:
        b1k[wdn['valves']] = -wdn['n_exp'][wdn['valves']] * b1k[wdn['valves']]
    b1k = np.tile(b1k, (1, q.shape[1])) * np.abs(q) ** (wdn['n_exp'] - 1)

    b2k = (-wdn['n_exp'] * ResCoeff) / theta
    if 'valves' in wdn and len(wdn['valves']) > 0:
        b2k[wdn['valves']] = b2k[wdn['valves']] / (-wdn['n_exp'][wdn['valves']])
    b2k = np.tile(b2k, (1, q.shape[1])) * np.abs(q) ** (wdn['n_exp'] - 1) * q

    return a11k, b1k, b2k



