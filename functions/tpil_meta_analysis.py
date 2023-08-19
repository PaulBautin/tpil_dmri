
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
