#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from neuromaps.datasets import fetch_fslr
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import interpolate
import matplotlib


fslr_l = nib.load(fetch_fslr().midthickness.L)
fslr_r = nib.load(fetch_fslr().midthickness.R)
fslr_data = np.concatenate((fslr_l.agg_data()[0], fslr_r.agg_data()[0]))

# xi = np.linspace(x.min(), x.max(), 500)
# yi = np.linspace(y.min(), y.max(), 500)
# xi, yi = np.meshgrid(xi, yi)

# # Interpolate z values on the grid
# zi = griddata((x, y), z, (xi, yi), method='linear')

#################### first figure ################
# Generate contour plot with black lines
# plt.figure(figsize=(8, 6))
# #contour = plt.contour(xi, yi, zi, 50)
# # fig = plt.figure(figsize=(12, 12))
# # ax = fig.add_subplot(projection='3d')
# size=2
# cmap = matplotlib.cm.get_cmap('Blues_r')
# n=30
# r = np.linspace(0,1, n)
# for i in range(n):
#     print((10*i) + 30)
#     condition = (fslr_data[:,2] <= (85/n)* i + 5) & (fslr_data[:,2] >= (85/n)*i )
#     x, y, z = fslr_data[:,0][condition], fslr_data[:,1][condition], fslr_data[:,2][condition]
#     plt.scatter(x, y, c=cmap(r[i]), s=size)

# plt.axis('off')
# plt.gca().set_aspect('equal')
# plt.savefig('/home/pabaua/dev_tpil/vizualisation/demo.png', transparent=True)
# plt.show()


############### second figure ####################
# #plt.figure(figsize=(8, 6))

# condition = fslr_data[:,2] >=70
# x, y, z = fslr_data[:,0][condition], fslr_data[:,1][condition], fslr_data[:,2][condition]

# xi = np.linspace(x.min(), x.max(), 1000)
# yi = np.linspace(y.min(), y.max(), 1000)
# xi, yi = np.meshgrid(xi, yi)
# zi = griddata((x, y), z, (xi, yi), method='nearest')
# zi[zi>=70] = 100
# zi[zi<=70] = 0

# plt.imshow(zi)
# plt.show()

# # contour = plt.contour(xi,yi, zi, 5)
# # plt.colorbar(contour)
# # plt.axis('off')
# # plt.gca().set_aspect('equal')
# # plt.show()

# from skimage import measure
# from skimage import feature
# from skimage import filters

# contours = measure.find_contours(zi, 99)
# print(contours)
# #plt.imshow(contours)
# #plt.show()


# # Display the image and plot all contours found
# fig, ax = plt.subplots()
# #ax.imshow(zi, cmap=plt.cm.gray)

# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=1, c='r')

# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()



########### third figure ###########
from skimage import measure
fs_vol = nib.load('/home/pabaua/dev_tpil/data/Freesurfer/freesurfer_v1/fsaverage/mri/aparc+aseg.mgz')
fs_data = fs_vol.get_fdata()
fs_data[fs_data != 0] = 1

cmap = matplotlib.cm.get_cmap('Blues_r')
n=5
r = np.linspace(0,1, n)
for i in [120,140,160,180, 200]:
    contours = measure.find_contours(fs_data[:,i,:].T, 0.99)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1, c=cmap((i - 120)/100))
plt.axis('off')
plt.gca().set_aspect('equal')
plt.show()