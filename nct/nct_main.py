
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
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset
from nimare.meta.cbma.mkda import MKDADensity
from nimare.correct import FWECorrector
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


#TODO: add cognitive topographies from neurosynth or brainmap
#TODO: test different meta-analysis techniques
#TODO: test different time horizons
#TODO: look up the activation values (could be binary)


def plot_points(values):
    print(values['v1'])
    brainnetome_atlas = nib.load("/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results/sub-pl007_ses-v1/labels_in_mni.nii.gz")
    coords = plotting.find_parcellation_cut_coords(brainnetome_atlas)[:210]
    plotting.plot_markers(values['v1'], coords, title="Volume z-score per node (CLBP - control) v1", node_vmin=-2, node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()
    plotting.plot_markers(values['v2'], coords, title="Volume z-score per node (CLBP - control) v2", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()
    plotting.plot_markers(values['v3'], coords, title="Volume z-score per node (CLBP - control) v3", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()

def cortical_volumes(path_roi_clbp, path_roi_control):
    # Get the freesurfer files
    files_clbp_lh = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.volume.txt")
    files_control_lh = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    files_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.volume.txt")

    ## clbp lh
    df_clbp_lh = pd.concat(pd.read_csv(files_clbp_lh[i], sep='\t') for i in range(len(files_clbp_lh)))
    df_clbp_lh[['participant_id', 'session']] = df_clbp_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_clbp_lh_avg = df_clbp_lh.drop(["lh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()

    ## control lh
    df_control_lh = pd.concat(
        pd.read_csv(files_control_lh[i], sep='\t') for i in range(len(files_control_lh)))
    df_control_lh['lh.BN_Atlas.volume'] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_T1w', n=0).str[0]
    df_control_lh[['participant_id', 'session']] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_control_lh_avg = df_control_lh.drop(["lh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()
    df_control_lh_std = df_control_lh.drop(["lh.BN_Atlas.volume"], axis=1).groupby(['session', 'participant_id']).mean().groupby(['session']).std()

    # diff and zscore lh
    df_lh_diff = (df_clbp_lh_avg - df_control_lh_avg) / df_control_lh_avg
    df_lh_z_score = (df_clbp_lh_avg - df_control_lh_avg) / df_control_lh_std

    ## clbp rh
    df_clbp_rh = pd.concat(pd.read_csv(files_clbp_rh[i], sep='\t') for i in range(len(files_clbp_rh)))
    df_clbp_rh[['participant_id', 'session']] = df_clbp_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_clbp_rh_avg = df_clbp_rh.drop(["rh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()

    ## control rh
    df_control_rh = pd.concat(pd.read_csv(files_control_rh[i], sep='\t') for i in range(len(files_control_rh)))
    df_control_rh['rh.BN_Atlas.volume'] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_T1w', n=0).str[0]
    df_control_rh[['participant_id', 'session']] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_control_rh_avg = df_control_rh.drop(["rh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()
    df_control_rh_std = df_control_rh.drop(["rh.BN_Atlas.volume"], axis=1).groupby(['session', 'participant_id']).mean().groupby(['session']).std()

    # diff and zscore rh
    df_rh_diff = (df_clbp_rh_avg - df_control_rh_avg) / df_control_rh_avg
    df_rh_z_score = (df_clbp_rh_avg - df_control_rh_avg) / df_control_rh_std

    diff = {ses: [val for pair in zip(df_lh_diff.loc[ses], df_rh_diff.loc[ses]) for val in pair][:210] for ses in ['v1', 'v2', 'v3']}
    z_score = {ses: [val for pair in zip(df_lh_z_score.loc[ses], df_rh_z_score.loc[ses]) for val in pair][:210] for ses in ['v1', 'v2', 'v3']}
    return diff, z_score

def fetch_neurosynth_data(out_dir):
    ## import and save dataset from neurosynth
    if os.path.isfile(os.path.join(out_dir, "neurosynth_dataset.pkl.gz")):
        neurosynth_dset = Dataset.load(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"), compressed=True)
    else:
        files = fetch_neurosynth(
            data_dir=out_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
        # Note that the files are saved to a new folder within "out_dir" named "neurosynth".
        # pprint(files)
        neurosynth_db = files[0]
        neurosynth_dset = convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
        print(neurosynth_dset)
    return neurosynth_dset

def get_studies_by_terms(neurosynth_dset, term_list):
    def get_study_by_term(term):
        ids = neurosynth_dset.get_studies_by_label(labels=["terms_abstract_tfidf__"+term], label_threshold=0.001)
        dset = neurosynth_dset.slice(ids)
        return dset
    term_dict = {term: get_study_by_term(term) for term in term_list}
    return term_dict

def apply_meta_analysis(neurosynth_dset_by_term, out_dir):
    if os.path.isfile(os.path.join(out_dir, "meta_cres.npy")):
        cres = np.load(os.path.join(out_dir, "meta_cres.npy"), allow_pickle='TRUE').item()
    else:
        meta = MKDADensity()
        results = {k: meta.fit(v) for k, v in neurosynth_dset_by_term.items()}
        corr = FWECorrector(method="montecarlo", n_iters=10, n_cores=1)
        cres = {k: corr.transform(v) for k, v in results.items()}
        np.save(os.path.join(out_dir, "meta_cres.npy"), cres)
    return cres

def apply_atlas(cres_dict):
    brainnetome_atlas = nib.load(
        "/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results/sub-pl007_ses-v1/labels_in_mni.nii.gz")
    # strategy is the name of a valid function to reduce the region with
    roi = maskers.NiftiLabelsMasker(brainnetome_atlas, strategy='mean')
    cres_atlas = {k: roi.fit_transform(v.get_map("z_level-voxel_corr-FWE_method-montecarlo")) for k, v in cres_dict.items()}
    return cres_atlas


def get_parcellation():
    def label_extractor(img_yeo, data_yeo, i):
        data_yeo_copy = data_yeo.copy()
        data_yeo_copy[data_yeo_copy != i] = 0
        data_yeo_copy[data_yeo_copy == i] = 1
        img_yeo_1 = nib.Nifti1Image(data_yeo_copy, img_yeo.affine, img_yeo.header)
        return img_yeo_1

    brainnetome_atlas = nib.load(
        "/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results/sub-pl007_ses-v1/labels_in_mni.nii.gz")
    # strategy is the name of a valid function to reduce the region with
    roi = maskers.NiftiLabelsMasker(brainnetome_atlas, strategy='mean')
    networks = {1:'visual', 2:'somatomotor', 3:'dorsal attention', 4:'ventral attention', 5:'limbic', 6:'frontoparietal', 7:'default'}
    yeo = datasets.fetch_atlas_yeo_2011()
    img_yeo = nib.load(yeo.thick_7)
    data_yeo = img_yeo.get_fdata()
    img_dict = {i: label_extractor(img_yeo, data_yeo, i) for i in np.delete(np.unique(data_yeo), 0)}
    dict_signal = {networks[k]: roi.fit_transform(v)[:,:210] for k, v in img_dict.items()}
    #img_signal = {k: roi.inverse_transform(v) for k, v in dict_signal.items()}
    return dict_signal

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


def plot_heatmap(x_con, x_clbp):
    # compute energy transition for each subject
    x_con = x_con[1].apply(lambda x: integrate_u(x)).apply(lambda x: np.sum(x))
    x_clbp = x_clbp[1].apply(lambda x: integrate_u(x)).apply(lambda x: np.sum(x))
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 3))
    df_g1 = (x_con.groupby(['x0', 'xf']).mean() / max(x_con.groupby(['x0', 'xf']).mean().tolist()))
    g1 = sns.heatmap(df_g1.unstack(level=-1),annot=True, ax=ax[0], cmap='Blues', square=True, vmin=0, vmax=1)
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=8)
    g1.set_xticklabels(g1.get_xticklabels(), rotation=45, fontsize=8)
    ax[0].set(ylabel='', xlabel='', title='transition energy for controls')
    df_g2 = (x_clbp.groupby(['x0', 'xf']).mean() / max(x_clbp.groupby(['x0', 'xf']).mean().tolist()))
    g2 = sns.heatmap(df_g2.unstack(level=-1), annot=True, ax=ax[1], cmap='Blues', square=True, vmin=0, vmax=1)
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=8)
    g2.set_xticklabels(g2.get_xticklabels(), rotation=45, fontsize=8)
    ax[1].set(ylabel='', xlabel='', title='transition energy for CLBPs')
    df_zscore = (x_clbp.groupby(['x0', 'xf']).mean() - x_con.groupby(['x0', 'xf']).mean()) / x_con.groupby(['x0', 'xf']).std()
    g3 = sns.heatmap(df_zscore.unstack(level=-1), annot=True, ax=ax[2], vmin=-2, vmax=2, cmap='RdBu', square=True)
    g3.set_yticklabels(g3.get_yticklabels(), rotation=0, fontsize=8)
    g3.set_xticklabels(g3.get_xticklabels(), rotation=45, fontsize=8)
    ax[2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control)')
    plt.show()


def main():
    """
    main function, gather stats and call plots
    """
    out_dir = os.path.abspath("/home/pabaua/example_data/example_data/")
    os.makedirs(out_dir, exist_ok=True)

    path_roi_clbp = '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    path_roi_control = '/home/pabaua/dev_tpil/data/Freesurfer/23_02_09_control_freesurfer_output'

    diff, z_score = cortical_volumes(path_roi_clbp, path_roi_control)
    #plot_points(z_score)

    # neurosynth_dset = fetch_neurosynth_data(out_dir)
    # term_list = ["pain", "fear", "memory", "default", "emotional", "attention", "sensory", "stress", "anxiety", "depression"]
    # neurosynth_dset_by_term = get_studies_by_terms(neurosynth_dset, term_list)
    # cres_dict = apply_meta_analysis(neurosynth_dset_by_term, out_dir)

    # cres_atlas = apply_atlas(cres_dict)
    cres_atlas = get_parcellation()

    # get connectivity and apply network control theory
    A = load_connectivity()
    A_con_norm, A_clbp_norm = normalize_connectomes(A)
    #x_con_v1, x_clbp_v1 = get_state_traj(A_con_norm['v1'], A_clbp_norm['v1'], cres_atlas, out_dir, vol_corr=diff['v1'], ses='v1_')
    x_con_v2, x_clbp_v2 = get_state_traj(A_con_norm['v2'], A_clbp_norm['v2'], cres_atlas, out_dir,vol_corr=diff['v2'], ses='v2_')
    #x_con_v3, x_clbp_v3 = get_state_traj(A_con_norm['v3'], A_clbp_norm['v3'], cres_atlas, out_dir,vol_corr=diff['v3'], ses='v3_')
    plot_heatmap(x_con_v2, x_clbp_v2)

if __name__ == "__main__":
    main()