# Prerequis: environnement virtuel avec python, pandas, numpy et matplotlib (env_tpil)
#
#########################################################################################


# Parser
#########################################################################################


import pandas as pd
import numpy as np
import os
import glob

def load_connectivity():
    # initialize A matrix which is the structural connectivity matrix NxN with N the number of nodes (n_nodes)
    A_con_v1 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v1.npy")
    A_con_v2 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v2.npy")
    A_con_v3 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v3.npy")

    A_con_v1 = np.array(np.vsplit(A_con_v1, 24))[:, :210, :210]
    A_con_v2 = np.array(np.vsplit(A_con_v2, 25))[:, :210, :210]
    A_con_v3 = np.array(np.vsplit(A_con_v3, 25))[:, :210, :210]



    A_clbp_v1 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v1.npy")
    A_clbp_v2 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v2.npy")
    A_clbp_v3 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v3.npy")

    A_clbp_v1 = np.array(np.vsplit(A_clbp_v1, 27))[:, :210, :210]
    A_clbp_v2 = np.array(np.vsplit(A_clbp_v2, 25))[:, :210, :210]
    A_clbp_v3 = np.array(np.vsplit(A_clbp_v3, 23))[:, :210, :210]

    A = {'v1':[A_con_v1, A_clbp_v1], 'v2':[A_con_v2, A_clbp_v2], 'v3':[A_con_v3, A_clbp_v3]}
    return A


def find_files_with_common_name(directory, common_name, labels_list=None):
    file_paths = glob.glob(directory + '/*/Compute_Connectivity/' + common_name)
    n = range(len(file_paths))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(file_paths[i]))) : pd.read_csv(file_paths[i], header=None, names=labels_list).set_index(labels_list) for i in n}
    df_paths = pd.concat(dict_paths)
    df_paths = df_paths.reset_index().rename(columns={'level_0': 'participant_id', 'level_1': 'roi'})
    # df_paths = df_paths[df_paths['participant_id'].str.contains('_ses-v1')]
    df_paths[['subject', 'session']] = df_paths['participant_id'].str.rsplit('_ses-', n=1, expand=True)
    df_paths = df_paths.drop("participant_id", axis=1)
    return df_paths