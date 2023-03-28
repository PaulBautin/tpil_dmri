# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from netneurotools.utils import get_centroids
import matplotlib.pyplot as plt



# Find center of mass of each image
brainnetome = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
centroids = get_centroids(brainnetome)

fig = plt.figure(figsize=(11,5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
ax.view_init(elev=0, azim=180)
ax.axis('off')
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
ax2.view_init(elev=90, azim=-90)
ax2.axis('off')
fig.colorbar(scatter2)
fig.tight_layout()
plt.show()


