MRI White-Matter Segmentation
================================================================

SPM and VBM
-----------

Voxel-based morphometry (VBM) involves a voxel-wise comparison of the
local concentration of gray matter between two groups of subjects. The
procedure is relatively straightforward and involves spatially
normalizing high-resolution images from all the subjects in the study
into the same stereotactic space. This is followed by segmenting the
gray matter from the spatially normalized images and smoothing the
gray-matter segments. Voxel-wise parametric statistical tests which
compare the smoothed gray-matter images from the two groups are
performed. Corrections for multiple comparisons are made using the
theory of Gaussian random fields. This paper describes the steps
involved in VBM, with particular emphasis on segmenting gray matter from
MR images with nonuniformity artifact. We provide evaluations of the
assumptions that underpin the method, including the accuracy of the
segmentation and the assumptions made about the statistical distribution
of the data. `(Ashburner,
2000) <https://doi.org/10.1006/nimg.2000.0582>`__

Strengths

-  Fully automated & quick
-  Investigates whole brain

Problems [Bookstein 2001, Davatzikos 2004, Jones 2005]

-  Alignment difficult; smallest systematic shifts between groups can be
   incorrectly interpreted as FA change
-  Needs smoothing to help with registration problems
-  No objective way to choose smoothing extent

TBSS
----

Solve alignment using alignment-invariant features, a skeleton.

1. Use medium-DoF nonlinear reg to pre-align all subjects’ FA Register
   FA images together to create average FA image. (nonlinear reg: FNIRT)
2. Create mean FA image (no smoothing)
3. “Skeletonise” Mean FA
4. Threshold Mean FA Skeleton. giving “objective” tract map
5. For each subject’s warped FA, fill each point on the mean-space
   skeleton with nearest maximum FA value (i.e., from the centre of the
   subject’s nearby tract)
6. Do cross-subject voxelwise stats on skeleton-projected FA
7. Threshold, (e.g., permutation testing, including multiple comparison
   correction)

atlas-based approach
--------------------
ROI to ROI-based approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

tractography and fiber clustering
---------------------------------
