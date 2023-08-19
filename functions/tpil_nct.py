
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Network control theory control vs. chronic pain patients
#
# example: python nct_main.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
from nilearn.plotting import plot_stat_map
import nibabel as nib
from nilearn import plotting
from nilearn import maskers
from nctpy.utils import matrix_normalization
from nctpy.energies import sim_state_eq
import matplotlib.pyplot as plt
import seaborn as sns
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.metrics import ave_control
from nilearn import plotting
import pandas as pd
from nilearn import plotting, datasets
import glob
from os.path import dirname as up

def normalize_connectomes(A):
    system = 'continuous'  # option 'discrete'
    A_con_norm = {k: np.array(list(matrix_normalization(A=v[0][i,:,:], c=1, system=system) for i in range(v[0].shape[0]))) for k,v in A.items()}
    A_clbp_norm = {k: np.array(list(matrix_normalization(A=v[1][i,:,:], c=1, system=system) for i in range(v[1].shape[0]))) for k,v in A.items()}
    return A_con_norm, A_clbp_norm


def get_state_traj(A_con, A_clbp, cres_atlas, out_dir, vol_corr, ses='v1'):
    print(os.path.join(out_dir,ses+ "x_con.pkl"))
    system = 'continuous'  # option 'discrete'
    # define initial and target states as random patterns of activity
    cres_atlas = {k: v[0][..., np.newaxis] for k, v in cres_atlas.items()}
    # set parameters
    T = 1  # time horizon
    rho = 1  # mixing parameter for state trajectory constraint
    S = np.eye(A_con.shape[1])  # nodes in state trajectory to be constrained
    B_con = np.eye(A_con.shape[1])  # which nodes receive input, uniform full control set
    B_clbp = np.eye(A_con.shape[1]) + np.eye(A_con.shape[1]) * vol_corr

    if os.path.isfile(os.path.join(out_dir, ses+ "x_con.pkl")):
        x_con = pd.read_pickle(os.path.join(out_dir, ses+ "x_con.pkl"))
    else:
        x_con = pd.concat({k2:pd.concat({k:pd.DataFrame.from_dict({i:get_control_inputs(A_norm=A_con[i,:,:], T=T, B=B_con, x0=v2, xf=v, system=system, rho=rho, S=S) for i in range(A_con.shape[0])}, 'index') for k, v in cres_atlas.items()}) for k2, v2 in cres_atlas.items()})
        x_con = x_con.rename_axis(index=['x0', 'xf', 'subject'])
        x_con.to_pickle(os.path.join(out_dir,ses+  "x_con.pkl"))

    if os.path.isfile(os.path.join(out_dir,ses+  "x_clbp.pkl")):
        x_clbp = pd.read_pickle(os.path.join(out_dir,ses+  "x_clbp.pkl"))
    else:
        x_clbp = pd.concat({k2:pd.concat({k:pd.DataFrame.from_dict({i:get_control_inputs(A_norm=A_clbp[i,:,:], T=T, B=B_clbp, x0=v2, xf=v, system=system, rho=rho, S=S) for i in range(A_clbp.shape[0])}, 'index') for k, v in cres_atlas.items()}) for k2, v2 in cres_atlas.items()}).rename_axis(index={0:'x0',1:'xf'})
        x_clbp = x_clbp.rename_axis(index=['x0', 'xf', 'subject'])
        x_clbp.to_pickle(os.path.join(out_dir,ses+  "x_clbp.pkl"))
    return x_con, x_clbp
