from netneurotools.plotting import plot_fslr, plot_point_brain
from netneurotools.datasets import fetch_conte69
from netneurotools.networks import threshold_network, struct_consensus
from netneurotools.utils import get_centroids
from netneurotools import metrics
import nibabel as nib
import numpy as np
from mayavi import mlab
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import glob
from functions.tpil_load_data import load_matrices
import pandas as pd
from bct.utils.visualization import adjacency_plot_und
import seaborn as sns
from nilearn import plotting
import scipy
import neuromaps
from neuromaps import stats, points, datasets, transforms, nulls
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres, surface_plotting
from brainspace import mesh

pd.options.display.width = 0

def get_surfaces():
    lhlabels = nib.load(lhlabels).darrays[0].data
    rhlabels = nib.load(rhlabels).darrays[0].data
    lhvert, lhface = [d.data for d in nib.load(lhsurface).darrays]
    rhvert, rhface = [d.data for d in nib.load(rhsurface).darrays]
    points.make_surf_graph(lhvert, lhface)
    plot_point_brain(np.arange(rhvert.shape[0]), coords=rhvert, size=1)
    plt.show()


def plot_degree(network_1, size):
    degree_1 = metrics.degrees_und(network_1)
    plot_point_brain(data=degree_1, coords=centroids, cbar=True, views_orientation='horizontal', views_size=(5, 4), size=size-np.min(size))
    plt.show()
    plot_fslr(data=degree_1, lhlabel=lhlabels, rhlabel=rhlabels)
    mlab.show()
    
def get_centroids_l(lhsurface, rhsurface, lhlabels, rhlabels):
    lhsurf = nib.load(lhsurface).agg_data()[0]
    rhsurf = nib.load(rhsurface).agg_data()[0]
    surfaces = np.concatenate((lhsurf, rhsurf))
    lhlabels = nib.load(lhlabels).agg_data()
    rhlabels = nib.load(rhlabels).agg_data()
    labels = np.concatenate((lhlabels, rhlabels)).astype("float")
    centroids = []
    for lab in np.unique(labels):
        roi = np.atleast_2d(surfaces[labels == lab].mean(axis=0))
        roi = surfaces[np.argmin(cdist(surfaces, roi), axis=0)[0]]
        centroids.append(roi)
    return np.array(centroids)


def plot_matrix(matrix, coords=None):
    df = pd.read_csv("/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/BN_Atlas_246_LUT.txt", sep=" ")
    adjacency_plot_und(matrix, coor=coords)
    sns.heatmap(matrix, cmap="crest")#, yticklabels=df['Unknown'].values)
    plt.show()


def plot_network(adj, adj_filtered, coords):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # Identify edges in the network
    edges = np.where(adj > 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    print(edge_val.shape)
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        ax[0].plot([x1, x2], [y1, y2], c=c, alpha=0.5, zorder=0, linewidth=0.1)
    ax[0].scatter(centroids[:, 0],centroids[:, 1], s=15, c=np.arange(1, 247), alpha=0.8)
    ax[0].axis('off')
    ax[0].set_aspect('equal')

    adj = adj_filtered
    edges = np.where(adj > 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    print(edge_val.shape)
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        ax[1].plot([x1, x2], [y1, y2], c=c, alpha=0.5, zorder=0, linewidth=0.1)
    ax[1].scatter(centroids[:, 0], centroids[:, 1], s=15, c=np.arange(1, 247), alpha=0.8)
    ax[1].axis('off')
    ax[1].set_aspect('equal')
    plt.show()

def plot_network_3(adj, degree, coords):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # Identify edges in the network
    edges = np.where(adj > 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax[0].plot([x1, x2], [y1, y2], [z1, z2], c=c, alpha=0.5, zorder=0, linewidth=0.1)
    ax[0].scatter(coords[:, 0], coords[:, 1], s=degree - np.min(degree), c=np.arange(1, 247),
                         alpha=0.8)
    ax[0].set_aspect('equal')
    ax[0].axis('off')

    # Identify edges in the network
    edges = np.where(adj > 0)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax[1].plot([x1, x2], [y1, y2], [z1, z2], c=c, alpha=0.5, zorder=0, linewidth=0.1)
    ax[1].scatter(coords[:, 0], coords[:, 1], s=degree - np.min(degree), c=np.arange(1, 247),
                            alpha=0.8)
    ax[1].axis('off')
    ax[1].set_aspect('equal')
    #fig.colorbar(ax[1])
    fig.tight_layout()
    plt.show()


def plot_network_2(adj, degree, coords):
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121, projection='3d')
    # Identify edges in the network
    edges = np.where(adj > 0)
    print(edges[0].shape)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c=c, alpha=0.5, zorder=0, linewidth=1)
    scatter = ax.scatter(coords[:, 0],coords[:, 1],coords[:, 2], s=degree-np.min(degree), c=np.arange(1, 247), alpha=0.8)
    ax.view_init(elev=0, azim=180)
    ax.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    ax.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    # Identify edges in the network
    edges = np.where(adj > 0)
    edge_cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=np.min(adj[edges]), vmax=np.max(adj[edges]))
    edge_val = edge_cmap(norm(adj[edges]))
    # Plot the edges
    for edge_i, edge_j, c in zip(edges[0], edges[1], edge_val):
        x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
        y1, y2 = coords[edge_i, 1], coords[edge_j, 1]
        z1, z2 = coords[edge_i, 2], coords[edge_j, 2]
        ax2.plot([x1, x2], [y1, y2], [z1, z2], c=c, alpha=0.5, zorder=0, linewidth=1)
    scatter_2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=degree-np.min(degree), c=np.arange(1, 247), alpha=0.8)
    ax2.view_init(elev=90, azim=-90)
    ax2.axis('off')
    ax2.set_box_aspect((np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])))
    fig.colorbar(scatter_2)
    fig.tight_layout()
    plt.show()

# Load surface and data
lhsurface, rhsurface = datasets.fetch_atlas(atlas='fsLR', density='32k')['midthickness'].L, datasets.fetch_atlas(atlas='fsLR', density='32k')['midthickness'].R
lhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii"
rhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii"
atlas_labels = np.concatenate((nib.load(lhlabels).agg_data(), nib.load(rhlabels).agg_data())).astype("float")

# Load demographics
df_demographics = pd.read_csv("/home/pabaua/Documents/tpil/CLBP_demographics.csv").drop(columns='Unnamed: 0')
df_demographics = df_demographics.rename(columns={'SubjID (subjectID)': 'subject', 'Age (in years)': 'AGE_AT_SCAN', 'Dx (diagnosis, 0=controls, 1=patients)': 'Dx', 'Sex (0=males, 1=females)': 'SEX'})
fun = lambda x: 'sub-' + str(x.lower().split('pt_')[1])
df_demographics['subject'] = df_demographics['subject'].apply(fun)

# Load connectivity matrices
filter = np.load("/home/pabaua/dev_tpil/results/results_connectflow/test/out_mask_1.npy")
list_clbp = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/clbp/**/Compute_Connectivity/commit2_weights.npy')
list_control = glob.glob('/home/pabaua/dev_tpil/data/22-11-16_connectflow/control/**/Compute_Connectivity/commit2_weights.npy')
df_connectivity = load_matrices(list_clbp, list_control)
df_connectivity = df_demographics.merge(df_connectivity)

# CLBP connectomes
df_clbp = df_connectivity.loc[df_connectivity['Dx'] == 1]
conn_clbp = np.dstack(df_clbp['connectivity'].values)

# Contol connectomes
df_control = df_connectivity.loc[df_connectivity['Dx'] == 0]
conn_control = np.dstack(df_control['connectivity'].values)

# Find center of mass of each image
brainnetome = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
centroids = get_centroids(brainnetome)

# plot_point_brain(data=np.arange(246), coords=centroids, cbar=True, views_orientation='horizontal', views_size=(5, 5))
# plt.show()
# hemiid = np.arange(1, 247) % 2  0 on rigtht side and 1 on left side
# plot_point_brain(data=hemiid, coords=centroids, cbar=True, views_orientation='horizontal', views_size=(5, 4))


hemiid = np.arange(1, 247) % 2
eu_distance = squareform(pdist(centroids, metric="euclidean"))

consensus_clbp = struct_consensus(conn_clbp, distance=eu_distance, hemiid=hemiid.reshape(-1, 1))
df_clbp['connectivity'] = df_clbp['connectivity'].apply(lambda x: x * consensus_clbp)
conn_clbp_filtered = np.dstack(df_clbp['connectivity'].values)

consensus_control = struct_consensus(conn_control, distance=eu_distance, hemiid=hemiid.reshape(-1, 1))
df_control['connectivity'] = df_control['connectivity'].apply(lambda x: x * consensus_control)
conn_control_filtered = np.dstack(df_control['connectivity'].values)

# Compare filtering
network_1 = np.mean(conn_clbp_filtered, where=conn_clbp>0, axis=2)
plot_network_3(network_1, metrics.degrees_und(network_1), centroids)
plt.show()
print("average path network 1: {}".format(np.sum(threshold_network(network_1, 28) * eu_distance) / 246))
network_1 = threshold_network(network_1, 28) * network_1
network_2 = np.mean(conn_clbp_filtered, where=conn_clbp_filtered>0, axis=2)
print("average path network 2: {}".format(np.sum((network_2 > 0) * eu_distance) / 246))
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0] = sns.heatmap(network_1, ax=ax[0])
# ax[1] = sns.heatmap(network_2, ax=ax[1])
# plt.show()

# degree distribution
#plot_degree(network_1, size=metrics.degrees_und(network_1))
#plot_degree(network_2)


network_NAC_1 = np.zeros([246, 246])
network_NAC_1[223, :] = network_1[223, :]
network_NAC_2 = np.zeros([246, 246])
network_NAC_2[223, :] = network_2[223, :]
plot_network_2(network_NAC_1, metrics.degrees_und(network_1), centroids)


#commnunicability_1 = metrics.communicability_wei(network_1)
#commnunicability_2 = metrics.communicability_wei(network_2)
#plot_network(commnunicability_1, commnunicability_2, centroids)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0] = sns.heatmap(degrees_clbp, ax=ax[0])
# ax[1] = sns.heatmap(degrees_control, ax=ax[1])
# plt.show()
degrees_clbp = metrics.degrees_und(conn_clbp_filtered)
degrees_control = metrics.degrees_und(conn_control_filtered)
z_score = (np.mean(degrees_clbp, axis=1) - np.mean(degrees_control, axis=1))/ np.std(degrees_clbp, axis=1)

brainnetome_data = brainnetome.get_fdata().copy()
brainnetome_data[brainnetome_data > 210] = 0
for i in range(210):
    brainnetome_data[brainnetome_data == i+1] = z_score[i]
z_score_img = nib.Nifti1Image(brainnetome_data, brainnetome.affine, brainnetome.header)
#rotated = nulls.burt2020(z_score_img, atlas='mni152', density='1mm', n_perm=2, seed=1234, n_proc=4)
#print(rotated.shape)


lhsurf = mesh.mesh_io.read_surface(str(lhsurface))
rhsurf = mesh.mesh_io.read_surface(str(rhsurface))
z_score_surf = transforms.mni152_to_fslr(z_score_img)
z_score_surf = np.concatenate((z_score_surf[0].agg_data(), z_score_surf[1].agg_data())).astype("float")
#plot_hemispheres(lhsurf, rhsurf, z_score_surf, color_bar=True)

print(datasets.available_tags())
nsynth = datasets.fetch_annotation(source='raichle', desc='cbf', space='fsLR')
nsynth = transforms.fslr_to_fslr(nsynth, target_density='32k')
nsynth_labels = np.concatenate((nsynth[0].agg_data(), nsynth[1].agg_data())).astype("float")
#plot_hemispheres(lhsurf, rhsurf, nsynth_labels, color_bar=True)
lhlabels = neuromaps.images.load_gifti(lhlabels)
rhlabels = neuromaps.images.load_gifti(lhlabels)
rotated = nulls.alexander_bloch(data=z_score_surf, atlas='fsLR', density='32k', n_perm=100, seed=1234)
print(rotated)
print(z_score_surf.shape)
print(nsynth_labels.shape)
corr, pvalue = stats.compare_images(z_score_surf, nsynth_labels, nulls=rotated)
print(f'r = {corr:.3f}, p = {pvalue:.3f}')


# pval = scipy.stats.ttest_ind(conn_clbp, conn_control, axis=2, nan_policy='propagate')[1]
# plot_fslr(data=pval[222], lhlabel=lhlabels, rhlabel=rhlabels)
# mlab.show()
#
# plot_matrix(threshold_network(eu_distance, retain=1), coords=centroids)
# plotting.plot_connectome(eu_distance, centroids, edge_threshold='90%')
# plt.show()