#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import image, surface, datasets, signal, connectome, plotting, interfaces
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from neuromaps.datasets import fetch_fslr
from neuromaps import nulls
from neuromaps import stats as stats_neuromaps
from surfplot import Plot, utils
from scipy import stats
from scipy import ndimage
from PIL import Image
import os
import seaborn as sns
from brainspace.gradient import GradientMaps
from brainspace import utils
import brainspace.datasets
import glob
import pandas as pd
import bct.algorithms as bct_alg
import bct.utils as bct


from netneurotools import metrics, networks


import bct
import numpy as np
from tqdm import tqdm
from sklearn.utils import check_random_state

def strength_preserving_rand(A, rewiring_iter = 10, nstage = 100, niter = 10000,
                             temp = 1000, frac = 0.5,
                             energy_func = None, energy_type = 'euclidean',
                             connected = None, verbose = False, seed = None):

    """
    Degree- and strength-preserving randomization of
    undirected, weighted adjacency matrix A
    Parameters
    ----------
    A : (N, N) array-like
        Undirected symmetric weighted adjacency matrix
    rewiring_iter : int, optional
        Rewiring parameter (each edge is rewired approximately maxswap times).
        Default = 10.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'euclidean': Sum of squares between strength sequence vectors
                         of the original network and the randomized network
            'max': The single largest value
                   by which the strength sequences deviate
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'euclidean'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Maintain connectedness of randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Print status to screen at the end of every stage. Default = False.
    seed: float, optional
        Random seed. Default = None.
    Returns
    -------
    B : (N, N) array-like
        Randomized adjacency matrix
    min_energy : float
        Minimum energy obtained by annealing
    Notes
    -------
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate adjacency matrix, B, with the same size, density, degree sequence,
    and weight distribution as A. The weights are then permuted to optimize the
    match between the strength sequences of A and B using simulated annealing.
    References
    -------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.

    2014-2022
    Richard Betzel, Indiana University
    Filip Milisav, McGill University

    Modification History:
    2014: Original (Richard Betzel)
    2022: Python translation, added connectedness-preservation functionality,
          new predefined energy types, and
          user-provided energy callable functionality (Filip Milisav)
    """

    try:
        A = np.array(A)
    except ValueError as err:
        msg = ('A must be array_like. Received: {}.'.format(type(A)))
        raise TypeError(msg) from err

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis = 1) #strengths of A

    if connected is None:
        connected = False if bct.number_of_components(A) > 1 else True

    #Maslov & Sneppen rewiring
    if connected:
        B = bct.randmio_und_connected(A, rewiring_iter, seed = seed)[0]
    else:
        B = bct.randmio_und(A, rewiring_iter, seed = seed)[0]

    u, v = np.triu(B, k = 1).nonzero() #upper triangle indices
    wts = np.triu(B, k = 1)[(u, v)] #upper triangle values
    m = len(wts)
    sb = np.sum(B, axis = 1) #strengths of B

    if energy_func is not None:
        energy = energy_func(s, sb)
    elif energy_type == 'euclidean':
        energy = np.sum((s - sb)**2)
    elif energy_type == 'max':
        energy = np.max(np.abs(s - sb))
    elif energy_type == 'mae':
        energy = np.mean(np.abs(s - sb))
    elif energy_type == 'mse':
        energy = np.mean((s - sb)**2)
    elif energy_type == 'rmse':
        energy = np.sqrt(np.mean((s - sb)**2))
    else:
        msg = ("energy_type must be one of 'euclidean', 'max', "
               "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type))
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print('\ninitial energy {:.5f}'.format(energy))

    for istage in tqdm(range(nstage), desc = 'annealing progress'):

        naccept = 0
        for i in range(niter):

            #permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime = sb.copy()
            sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
            sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime, s)
            elif energy_type == 'euclidean':
                energy_prime = np.sum((sb_prime - s)**2)
            elif energy_type == 'max':
                energy_prime = np.max(np.abs(sb_prime - s))
            elif energy_type == 'mae':
                energy_prime = np.mean(np.abs(sb_prime - s))
            elif energy_type == 'mse':
                energy_prime = np.mean((sb_prime - s)**2)
            elif energy_type == 'rmse':
                energy_prime = np.sqrt(np.mean((sb_prime - s)**2))
            else:
                msg = ("energy_type must be one of 'euclidean', 'max', "
                       "'mae', 'mse', or 'rmse'. "
                       "Received: {}.".format(energy_type))
                raise ValueError(msg)

            #permutation acceptance criterion
            if (energy_prime < energy or
               rs.rand() < np.exp(-(energy_prime - energy)/temp)):
                sb = sb_prime.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        #temperature update
        temp = temp*frac
        if verbose:
            print('\nstage {:d}, temp {:.5f}, best energy {:.5f}, '
                  'frac of accepted moves {:.3f}'.format(istage, temp,
                                                         energymin,
                                                         naccept/niter))

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin


def _ecdf(data):
    """
    Estimate empirical cumulative distribution function of `data`.

    Taken directly from StackOverflow. See original answer at
    https://stackoverflow.com/questions/33345780.

    Parameters
    ----------
    data : array_like

    Returns
    -------
    prob : numpy.ndarray
        Cumulative probability
    quantiles : numpy.darray
        Quantiles
    """
    sample = np.atleast_1d(data)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    prob = np.cumsum(counts).astype(float) / sample.size

    # match MATLAB
    prob, quantiles = np.append([0], prob), np.append(quantiles[0], quantiles)

    return prob, quantiles

def struct_consensus(data, distance, weighted=False):
    num_node, _, num_sub = data.shape      # info on connectivity matrices
    pos_data = data > 0                    # location of + values in matrix
    pos_data_count = pos_data.sum(axis=2)  # num sub with + values at each node

    with np.errstate(divide='ignore', invalid='ignore'):
        average_weights = data.sum(axis=2) / pos_data_count

    # empty array to hold inter/intra hemispheric connections
    consensus = np.zeros((num_node, num_node))

    keep_conn = distance > 0

    # mask the distance array for only those edges we want to examine
    full_dist_conn = distance * keep_conn
    upper_dist_conn = np.atleast_3d(np.triu(full_dist_conn))

    # generate array of weighted (by distance), positive edges across subs
    pos_dist = pos_data * upper_dist_conn
    pos_dist = pos_dist[np.nonzero(pos_dist)]

    # determine average # of positive edges across subs
    # we will use this to bin the edge weights
    avg_conn_num = len(pos_dist) / num_sub

    # estimate empirical CDF of weighted, positive edges across subs
    cumprob, quantiles = _ecdf(pos_dist)
    cumprob = np.round(cumprob * avg_conn_num).astype(int)

    # empty array to hold group-average matrix for current connection type
    # (i.e., inter/intra hemispheric connections)
    group_conn_type = np.zeros((num_node, num_node))

    # iterate through bins (for edge weights)
    for n in range(1, int(avg_conn_num) + 1):
        # get current quantile of interest
        curr_quant = quantiles[np.logical_and(cumprob >= (n - 1),
                                                cumprob < n)]
        if curr_quant.size == 0:
            continue

        # find edges in distance connectivity matrix w/i current quantile
        mask = np.logical_and(full_dist_conn >= curr_quant.min(),
                                full_dist_conn <= curr_quant.max())
        i, j = np.where(np.triu(mask))  # indices of edges of interest

        c = pos_data_count[i, j]   # get num sub with + values at edges
        w = average_weights[i, j]  # get averaged weight of edges

        # find locations of edges most commonly represented across subs
        indmax = np.argwhere(c == c.max())

        # determine index of most frequent edge; break ties with higher
        # weighted edge
        if indmax.size == 1:  # only one edge found
            group_conn_type[i[indmax], j[indmax]] = 1
        else:                 # multiple edges found
            indmax = indmax[np.argmax(w[indmax])]
            group_conn_type[i[indmax], j[indmax]] = 1

    consensus[:, :] = group_conn_type

    # collapse across hemispheric connections types and make symmetrical array
    #consensus = consensus.sum(axis=2)
    consensus = np.logical_or(consensus, consensus.T).astype(int)

    if weighted:
        consensus = consensus * np.mean(data, axis=2)
    return consensus




def random_wiring(W, L, its=10):
    conn_rand = np.array([networks.match_length_degree_distribution(W, L, nbins=1, nswap=215*20)[1] for i in range(its)])
    return conn_rand

def zscore_adj(W, W_rand):
    mean_conn_rand = np.mean(W_rand, axis=0, dtype=np.longdouble)
    print(mean_conn_rand)
    std_conn_rand = np.std(W_rand, axis=0, dtype=np.longdouble)
    print(std_conn_rand)
    zscore = ((W - mean_conn_rand) / std_conn_rand)
    #zscore = np.divide((W - mean_conn_rand), std_conn_rand, where=(std_conn_rand > 0))
    return zscore



def find_files_with_common_name(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df = df[~df.subject.isin(['sub-10'])]
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df


def find_files_with_common_name_rapidtide(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(fp)) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df = df[~df.subject.isin(['sub-10'])]
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df


def find_files_with_common_name_structural(directory, common_name, id='conn'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))): fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'participant_id', 0: 'path_'+id})
    df = df[df['participant_id'].str.contains('_ses-v1')]
    df[['subject', 'session']] = df['participant_id'].str.rsplit('_ses-', n=1, expand=True)
    df[['subject', 'num']] = df['subject'].str.rsplit('pl0', n=1, expand=True)
    df['subject'] = df['subject'] + df['num']
    df = df.drop(["participant_id", 'session', 'num'], axis=1)
    df['img_'+id] = df['path_'+id].apply(lambda x: pd.read_csv(x, header=None).values)
    return df


def compute_mean_first_passage_time(df_connectivity_matrix):
    df = df_connectivity_matrix.apply(lambda x: bct_alg.mean_first_passage_time(x))
    df = df.apply(lambda x: bct.invert(x))
    std = np.std(np.array(df.values.tolist()), axis=0)
    #df = df_connectivity_matrix.groupby('subject').apply(lambda x: bct_alg.mean_first_passage_time(bct.invert(x.drop(['subject', 'roi'], axis=1).to_numpy())))
    return df.mean(), std


def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    slices = 575

    # Load images into dataframe
    directory = '/media/pabaua/Transcend/fmriprep/23-10-19/V1/'
    #common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    #df_func_img = find_files_with_common_name(directory, common_name, id='func')
    directory_rapidtide = '/home/pabaua/dev_tpil/data/rapidtide/'
    directory_conn = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/all_schaefer/'
    common_name = '*/*_task-rest_space-MNI152NLin6Asym_desc-lfofilterCleaned_bold.nii.gz'
    df_func_img = find_files_with_common_name_rapidtide(directory_rapidtide, common_name, id='func')
    common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-aparcaseg_dseg.nii.gz'
    df_seg_img = find_files_with_common_name(directory, common_name, id='seg')
    common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    df_mask_img = find_files_with_common_name(directory, common_name, id='mask')
    df_dem = pd.read_excel('/home/pabaua/dev_tpil/data/Données_Paul_v2.xlsx', sheet_name=1)
    df_img = pd.merge(df_func_img, df_seg_img, on='subject')
    df_img = pd.merge(df_img, df_mask_img, on='subject')
    df_img = pd.merge(df_img, df_dem, on='subject')
    df_img = df_img[df_img.subject.isin(['sub-06', 'sub-12', 'sub-07', 'sub-02'])] # , 'sub-07', 'sub-02', 'sub-04', 'sub-06', 'sub-14', 'sub-15'
    print(df_img)

    # load segmentations    sc_rand = strength_preserving_rand(sc_ref)
    seg_img = nib.load('/home/pabaua/dev_tpil/data/sub-07-2570132/sub-07-2570132/sub-07/anat/fsl_first/sub-07_all_fast_firstseg.nii.gz')
    #seg_img = nib.load('/home/pabaua/dev_tpil/data/Mahsa 2/Mahsa/Step1_1/cluster_ant_BL_R.nii.gz')
    mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    #test = image.resample_to_img(mask_img, df_img[df_img.subject == 'sub-12'].img_func.values[0].slicer[:,:,:,30], interpolation='nearest')
    #nib.save(test, '/home/pabaua/Downloads/cluster_ant_BL_R_paul.nii.gz')


    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    # print(df_img[df_img.subject == 'sub-12'].img_func.values)
    # plotting.plot_stat_map(atlas.maps, bg_img=df_img[df_img.subject == 'sub-12'].img_func.values[0].slicer[:,:,:,30])
    # plotting.plot_stat_map(mask_img, bg_img=df_img[df_img.subject == 'sub-12'].img_func.values[0].slicer[:,:,:,30])
    # plotting.plot_stat_map(df_img[df_img.subject == 'sub-12'].img_mask.values[0], bg_img=df_img[df_img.subject == 'sub-12'].img_func.values[0].slicer[:,:,:,30])
    #labels = pd.read_csv('/media/pabaua/Transcend/fmriprep/23-10-19/V1/sub-06/desc-aparcaseg_dseg.tsv', sep = '\t')
    #atlas_labels = labels[labels.index != 0].name.values
    #df_img['confounds'] =  df_img.apply(lambda x : interfaces.fmriprep.load_confounds(x.path_func, strategy=['ica_aroma']), axis=1)
    #print(df_img['confounds'])
    df_img['ext'] =  df_img.apply(lambda x : NiftiLabelsMasker(atlas.maps, labels=atlas.labels, mask_img=x.img_mask).fit_transform(x.img_func.slicer[:,:,:,:slices]), axis=1)
    df_img['ext_sub'] =  df_img.apply(lambda x : NiftiLabelsMasker(mask_img).fit_transform(x.img_func.slicer[:,:,:,:slices]), axis=1)
    df_img['clean']=  df_img.apply(lambda x : signal.clean(x.ext, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T, axis=1)
    df_img['clean_sub']=  df_img.apply(lambda x : signal.clean(x.ext_sub, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T, axis=1)
    df_img['clean_tot'] =  df_img.apply(lambda x : np.vstack((x.clean, x.clean_sub)), axis=1)
    df_img['adj'] =  df_img.apply(lambda x : np.arctanh(connectome.ConnectivityMeasure(kind='correlation', standardize='zscore_sample').fit_transform([x.clean_tot.T])[0]), axis=1)

    labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=200, join=True)
    mask = (labelling_schaefer != 0)
    df_img['surf_data_func'] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x.adj[-1, :-1], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
    df_img['rotated'] = df_img.apply(lambda x: nulls.alexander_bloch(x.surf_data_func, atlas='fsLR', density='32k', n_perm=100, seed=1234), axis=1)

    id='len'
    common_name = '*/Compute_Connectivity/len.csv'
    df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    df_img = pd.merge(df_img, df_conn_img, on='subject')

    id='sc'
    common_name = '*/Compute_Connectivity/sc_vol_normalized.csv'
    df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    df_img = pd.merge(df_img, df_conn_img, on='subject')
    sc_ref = struct_consensus(np.stack(df_img['img_' + id].to_numpy()).T, df_img.img_len.mean(), weighted=True)
    sc_rand = np.array([strength_preserving_rand(sc_ref)[0] for i in range(20)])
    df_img[id + '_rand'] = df_img.apply(lambda x: sc_rand, axis=1)

    id='commit2'
    common_name = '*/Compute_Connectivity/commit2_weights.csv'
    df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    df_img = pd.merge(df_img, df_conn_img, on='subject')
    commit2_ref = struct_consensus(np.stack(df_img['img_' + id].to_numpy()).T, df_img.img_len.mean(), weighted=True)
    #commit2_rand = random_wiring(commit2_ref, df_img.img_len.mean(), its=100)
    commit2_rand = np.array([strength_preserving_rand(commit2_ref)[0] for i in range(20)])
    df_img[id + '_rand'] = df_img.apply(lambda x: commit2_rand, axis=1)

    id='afd'
    common_name = '*/Compute_Connectivity/afd_fixel.csv'
    df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    df_img = pd.merge(df_img, df_conn_img, on='subject')
    afd_ref = struct_consensus(np.stack(df_img['img_' + id].to_numpy()).T, df_img.img_len.mean(), weighted=True)
    afd_rand = np.array([strength_preserving_rand(afd_ref)[0] for i in range(20)])
    df_img[id + '_rand'] = df_img.apply(lambda x: afd_rand, axis=1)

    id='len'
    len_ref = struct_consensus(np.stack(df_img['img_' + id].to_numpy()).T, df_img['img_' + id].mean(), weighted=True)
    len_rand = np.array([strength_preserving_rand(len_ref)[0] for i in range(20)])
    df_img[id + '_rand'] = df_img.apply(lambda x: len_rand, axis=1)

    df_img.to_pickle('/home/pabaua/dev_tpil/data/correlations_rewiring_pre.pkl')
    df_img = pd.read_pickle('/home/pabaua/dev_tpil/data/correlations_rewiring_pre.pkl')

    labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=200, join=True)
    mask = (labelling_schaefer != 0)
    list_metrics = []
    for id in ['commit2', 'sc', 'afd', 'len']:
        # shortest path
        metric = '_sp'
        func = lambda x: bct_alg.distance_wei(bct.invert(x))[0]
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: np.array([func(slice_2d) for slice_2d in x[id + '_rand']]), axis=1)
        df_img['img_' + id + metric] = df_img.apply(lambda x: func(x['img_' + id]), axis=1)
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: (x['img_' + id + metric] - np.mean(x[id + '_rand'+ metric], axis=0)) / np.std(x[id + '_rand'+ metric], axis=0), axis=1)
        sns.heatmap(df_img[id + '_rand' + metric].mean(), cmap="coolwarm")
        plt.show()
        df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + '_rand' + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
        surfaces = fetch_fslr()
        lh, rh = surfaces['inflated']
        p = Plot(lh, rh, views=['lateral','medial'])
        p.add_layer(df_img[id + metric].mean(), cbar=True, cmap='coolwarm')
        fig = p.build()
        plt.show()
        surfaces = fetch_fslr()
        lh, rh = surfaces['inflated']
        p = Plot(lh, rh, views=['lateral','medial'])
        p.add_layer(df_img.surf_data_func.mean(), cbar=True, cmap='coolwarm')
        fig = p.build()
        plt.show()
        df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
        list_metrics += [id + metric]

        # # mean first passage time
        metric = '_mfpt'
        func = lambda x: bct_alg.mean_first_passage_time(x)
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: np.array([func(slice_2d) for slice_2d in x[id + '_rand']]), axis=1)
        df_img['img_' + id + metric] = df_img.apply(lambda x: func(x['img_' + id]), axis=1)
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: (x['img_' + id + metric] - np.mean(x[id + '_rand'+ metric], axis=0)) / np.std(x[id + '_rand'+ metric], axis=0), axis=1)
        df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + '_rand' + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
        df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
        list_metrics += [id + metric]

        # # mean first passage time
        metric = '_si'
        func = lambda x: bct_alg.search_information(x)
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: np.array([func(slice_2d) for slice_2d in x[id + '_rand']]), axis=1)
        df_img['img_' + id + metric] = df_img.apply(lambda x: func(x['img_' + id]), axis=1)
        df_img[id + '_rand' + metric] = df_img.apply(lambda x: (x['img_' + id + metric] - np.mean(x[id + '_rand'+ metric], axis=0)) / np.std(x[id + '_rand'+ metric], axis=0), axis=1)
        df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + '_rand' + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
        df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
        list_metrics += [id + metric]


    df_img.to_pickle('/home/pabaua/dev_tpil/data/correlations_rewiring.pkl')
    df_img = pd.read_pickle('/home/pabaua/dev_tpil/data/correlations_rewiring.pkl')

    sns.set_theme(style='whitegrid')
    sns.violinplot(data=df_img)
    plt.ylim((-0.7,0.7))
    plt.ylabel("pearson's r")
    plt.show()
    # print(df_img)
    # id='commit2'
    # common_name = '*/Compute_Connectivity/commit2_weights.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id='commit2')
    # df_img = pd.merge(df_img, df_conn_img, on='subject')



    # for id in ['sc', 'commit2']:
    #     labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=200, join=True)
    #     mask = (labelling_schaefer != 0)
    #     surf_data_func = utils.parcellation.map_to_labels(np.abs(df_img.adj.mean()[-1, :-1]), labelling_schaefer, mask=mask, fill=np.nan)
    #     rotated = nulls.alexander_bloch(surf_data_func, atlas='fsLR', density='32k', n_perm=100, seed=1234)

    #     df_img[id + 'sp'] = df_img.apply(lambda x: bct.invert(bct_alg.distance_wei(bct.invert(x['img_' + id]))[0]), axis=1)
    #     surf_data_sp = utils.parcellation.map_to_labels(np.abs(df_img[id + 'sp'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_sp, pval_sp, nulls_sp = stats_neuromaps.compare_images(surf_data_func, surf_data_sp, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_sp:.02f}, p = {pval_sp:.04f}')

    #     df_img[id + 'mfpt'] = df_img.apply(lambda x: bct.invert(bct_alg.mean_first_passage_time(x['img_' + id])), axis=1)
    #     surf_data_mfpt = utils.parcellation.map_to_labels(np.abs(df_img[id + 'mfpt'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_mfpt, pval_mfpt, nulls_mfpt = stats_neuromaps.compare_images(surf_data_func, surf_data_mfpt, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_mfpt:.02f}, p = {pval_mfpt:.04f}')

    #     df_img[id + 'si'] = df_img.apply(lambda x: bct_alg.search_information(x['img_' + id]), axis=1)
    #     surf_data_si = utils.parcellation.map_to_labels(np.abs(df_img[id + 'si'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_si, pval_si, nulls_si = stats_neuromaps.compare_images(surf_data_func, surf_data_si, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_si:.02f}, p = {pval_si:.04f}')
    #     dict = {id: [corr_sp, corr_mfpt, corr_si]}



    # communication_names = ['mfpt', 'shortest path', 'search information']
    # plt.figure(figsize = (7,7))
    # sns.set(style="ticks",font_scale=2)
    # p1 = sns.boxplot([nulls_mfpt, nulls_sp, nulls_si], color='white', orient='h')
    # ax = plt.gca()
    # ax.set_xlim([-0.2, 0.2])
    # ax.set_yticklabels(communication_names)
    # plt.scatter(y=range(len(communication_names)), x=[corr_mfpt, corr_sp, corr_si], s=50)
    # plt.legend(frameon=False)
    # plt.xlabel("pearson's r")
    # plt.ylabel("communication model")
    # plt.legend(frameon=False)
    # p1.xaxis.tick_top()
    # plt.tight_layout()
    # plt.show()




    # surf_data_func = utils.parcellation.map_to_labels(np.abs(df_img.commit2_rand.mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)

    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(surf_data_func, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()


    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(surf_data_mfpt, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()






    #mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    #ext_sub = NiftiLabelsMasker(seg_img).fit_transform(func_img.slicer[:,:,:,:slices])

    # # Define surfaces
    # inner_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.white.32k_fs_LR.surf.gii'
    # inner_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.white.32k_fs_LR.surf.gii'
    # surf_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.pial.32k_fs_LR.surf.gii'
    # surf_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.pial.32k_fs_LR.surf.gii'

    # # Process functional image and surface data
    # slices = func_img.get_fdata().shape[3]
    # #slices = 34
    # print(slices)

    # # Fetch surface atlas
    # atlas = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz')
    # atlas = datasets.fetch_atlas_schaefer_2018()
    # ext_cortex = NiftiLabelsMasker(atlas.maps, labels=atlas.labels).fit_transform(func_img.slicer[:,:,:,:slices])
    # #mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    # ext_sub = NiftiLabelsMasker(seg_img).fit_transform(func_img.slicer[:,:,:,:slices])

    # clean_cortex = signal.clean(ext_cortex, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    # clean_sub = signal.clean(ext_sub, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    # clean_parcels = np.vstack((clean_cortex, clean_sub))
    # print(clean_parcels.shape)

    # correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    # correlation_matrix = correlation_measure.fit_transform([clean_parcels.T])[0]
    # sns.heatmap(correlation_matrix)
    # plt.show()

    # # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    # # mask = labelling_schaefer != 0
    # # surf_data = utils.parcellation.map_to_labels(np.abs(correlation_matrix[-1,:]), labelling_schaefer, mask=mask, fill=np.nan)

    # # # Plot surfaces with functional data
    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # p.add_layer(surf_data, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show()




    # gm = GradientMaps(n_components=10, random_state=0)
    # gm.fit(correlation_matrix)
    # print(gm.gradients_[:,0].shape)

    # labelling_lh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # labelling_rh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    # grad = [None] * 2
    # mask = labelling_schaefer != 0
    # for i in range(2):
    #     # map the gradient to the parcels
    #     grad[i] = utils.parcellation.map_to_labels(gm.gradients_[:, i], labelling_schaefer, mask=mask, fill=np.nan)


    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(grad[0], cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()
    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(grad[1], cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()


    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # ax.scatter(range(gm.lambdas_.size), gm.lambdas_)
    # ax.set_xlabel('Component Nb')
    # ax.set_ylabel('Eigenvalue')
    # plt.show()

    # # # Compute correlation
    # # stat_map_lh = np.abs([stats.pearsonr(clean_sub, clean_lh[:,i])[0] for i in range(clean_lh.shape[1])])
    # # stat_map_rh = np.abs([stats.pearsonr(clean_sub, clean_rh[:,i])[0] for i in range(clean_rh.shape[1])])

    # # Plot surfaces with correlation data
    # #p = Plot(infl_lh, infl_rh, views=['lateral','medial'], zoom=1.2)
    # #stat_map_lh = utils.threshold(stat_map_lh, 0.3)
    # #stat_map_rh = utils.threshold(stat_map_rh, 0.3)
    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # p.add_layer({'left':stat_map_lh, 'right':stat_map_rh}, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show(block=True)

    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # lh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # # rh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data() 
    # # val_lh = np.zeros(stat_map_lh.shape[0])
    # # val_rh = np.zeros(stat_map_rh.shape[0])
    # # for i in np.arange(211):
    # #     val_lh += np.where(lh_BN == i, np.mean((lh_BN == i) * stat_map_lh), 0)
    # #     val_rh += np.where(rh_BN == i, np.mean((rh_BN == i) * stat_map_rh), 0)
    # # p.add_layer({'left':val_lh, 'right':val_rh}, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show(block=True)



    # # clean_lh = ndimage.gaussian_filter1d(clean_lh, axis=0, sigma=3)
    # # clean_rh = ndimage.gaussian_filter1d(clean_rh, axis=0, sigma=3)

    # # # Save each frame as an image
    # # frames = []
    # # for frame in range(surf_data.shape[0]):
    # #     p = Plot(infl_lh, infl_rh, views=['lateral', 'medial'], flip=True)
    # #     p.add_layer({'left':clean_lh[frame,:], 'right':clean_rh[frame,:]}, cbar=True, cmap='inferno', color_range=(-3,3), cbar_label=f"t = {np.round(frame* 1.075)} s")
    # #     fig = p.build()
    # #     filename = f"frame_{frame}.png"
    # #     plt.savefig(filename)
    # #     frames.append(filename)
    # #     plt.close(fig)

    # # # Create a GIF from the saved frames
    # # with Image.open(frames[0]) as img:
    # #     img.save("animation.gif", save_all=True, append_images=[Image.open(f) for f in frames[1:]], duration=100, loop=0)

    # # # Remove the individual frame images
    # # for frame in frames:
    # #     os.remove(frame)





if __name__ == "__main__":
    main()
