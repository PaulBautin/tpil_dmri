# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import glob
from brainspace.mesh import mesh_io


def load_data_xlsx(file, group_name):
    """
    Creates a dataframe based on information present on TractometryFlow output xlsx files
    :param file: xlsx file containing information for statistics. Ex: 'mean_std.xlsx'
    :return sheet_to_df_map: 3D dataframe containing information present in each sheet of xlsx file
    """
    # reads all sheets and stores dataframes in a dictionary
    df_dict = pd.read_excel(file, sheet_name=None, index_col=0)
    df = pd.concat(df_dict, axis=1).stack().rename_axis(['subject', 'tract']).reset_index()
    df[['tract', 'point']] = df['tract'].str.rsplit('_', n=1, expand=True)
    df[['subject', 'session']] = df['subject'].str.rsplit('_ses-', n=1, expand=True)
    df = df.drop(df.filter(regex='std', axis=1).columns, axis=1)
    #df.columns = df.columns.str.removesuffix("_metric_mean")
    df['group_name'] = group_name
    df['point'] = df['point'].apply(lambda x : int(x))
    return df


def load_data_xlsx_add(file, df_m, names=["subject", "tract", "session"]):
    """
    Creates a dataframe based on information present on TractometryFlow output xlsx files
    :param file: xlsx file containing information for statistics. Ex: 'mean_std.xlsx'
    :return sheet_to_df_map: 3D dataframe containing information present in each sheet of xlsx file
    """
    # reads all sheets and stores dataframes in a dictionary
    df_dict = pd.read_excel(file, sheet_name=None, index_col=0)
    df = pd.concat(df_dict, axis=1).stack().rename_axis(['subject', 'tract']).reset_index()
    df[['subject', 'session']] = df['subject'].str.rsplit('_ses-', 1, expand=True)
    df = df.set_index(names)
    df_m = df_m.set_index(names)
    df_m = df_m.combine_first(df)
    return df_m.reset_index()


def diff_metrics(df_con, df_clbp):
    # prepare dataframe for PCA computation
    df_con_mean = df_con.groupby(["session", "tract", "point"]).mean()
    df_clbp_mean = df_clbp.groupby(["session", "tract", "point"]).mean()
    return df_con_mean.subtract(df_clbp_mean).reset_index()


import numpy as np
import glob
import pandas as pd
from brainspace.mesh import mesh_io

def load_matrices(list_g1, list_g2, filter=1):
    matrices_g1 = {path.split('clbp/')[1].split('/Compute_Connectivity/')[0]: [np.load(path)[:-3,:-3] * filter] for path in list_g1}
    matrices_g2 = {path.split('control/')[1].split('/Compute_Connectivity/')[0]: [np.load(path)[:-3,:-3] * filter] for path in list_g2}
    df_matrices_g1 = pd.DataFrame.from_dict(matrices_g1, orient='index', columns=['connectivity']).reset_index()
    df_matrices_g2 = pd.DataFrame.from_dict(matrices_g2, orient='index', columns=['connectivity']).reset_index()
    df_matrices_g1[['subject', 'session']] = df_matrices_g1['index'].str.rsplit('_ses-', 1, expand=True)
    df_matrices_g2[['subject', 'session']] = df_matrices_g2['index'].str.rsplit('_ses-', 1, expand=True)
    df = pd.concat([df_matrices_g1.drop(columns='index'), df_matrices_g2.drop(columns='index')])
    return df


def load_surface():
    surf_lh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii')
    surf_rh = mesh_io.read_surface('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii')
    return surf_lh, surf_rh

def load_annot():
    # Brainnetome Atlas has 210 cortical and 36 subcortical subregions
    # vertices with no id have an id set to -1 (Example: subcortical regions)
    annot_lh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/lh.BN_Atlas.annot"
    annot_rh = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/label/rh.BN_Atlas.annot"
    atlas = np.concatenate((nib.freesurfer.read_annot(annot_lh)[0], nib.freesurfer.read_annot(annot_rh)[0]),
                           axis=0).astype(float)
    atlas[atlas <= 0] = np.nan
    return atlas




