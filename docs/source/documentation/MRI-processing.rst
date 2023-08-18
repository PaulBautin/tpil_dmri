Tractoflow
==========

MRI processing was done using TractoFlow, which is a fully automated and
reproducible diffusion MRI processing pipeline used to reconstruct the
whole brain white matter architecture.

How
---

TractoFlow first outputs diffusion metrics. 1. DTI metrics were computed
from the b=1000 mm2/s shell using ``scil_compute_dti_metrics.py`` which
is based on ``dipy.reconst.dti``
(`tutorial <https://dipy.org/documentation/1.5.0/examples_built/reconst_dti/#example-reconst-dti>`__).
The tensor fit method: weighted least squares (WLS) 2. fODF were
computed from b=2000 mm2/s (>50 direction -> HARDI) shell using
``scil_compute_ssst_fodf.py`` which is based on
``dipy.reconst.csdeconv``\ (`tutorial <https://dipy.org/documentation/1.5.0/examples_built/reconst_csd/#example-reconst-csd>`__).
The FRF were by default. fODF metrics were then computed using
``scil_compute_fodf_metrics.py`` see [Raffelt et al. NeuroImage 2012]
and [Dell’Acqua et al. HBM 2013] for the definitions. Tractoflow then
outputs tractograms 4. PFT tractogram are computed from fODF maps using
``scil_compute_pft.py`` based on
``dipy.tracking.local_tracking.ParticleFilteringTracking``
(`tutorial <https://dipy.org/documentation/1.5.0/examples_built/tracking_pft/#example-tracking-pft>`__).
We use a probabilistic tracking algorithm, the stopping criterion is
based on partial volume estimation (PVE) maps outputed by FSL-FAST and
the continuous map criterion (CMC). 5. Local tracking tractogram are
computed from fODF maps using ``scil_compute_local_tracking.py`` based
on ``dipy.tracking.local_tracking.LocalTracking``
(`tutorial <https://dipy.org/documentation/1.5.0/examples_built/tracking_probabilistic/#example-tracking-probabilistic>`__).
We use a probabilistic tracking algorithm, the stopping criterion is
based on a FA map treshold mask 5. Free Water maps were computed from
b=300 mm2/s and b=1000 mm2/s shells using ``scil_compute_freewater.py``
which is based on Accelerated microstructure imaging via convex
optimization (AMICO). Instructions can be found
`here <https://github.com/daducci/AMICO/wiki>`__

Robustness and reproducibility
------------------------------

leveraging Nextflow and Singularity for robustness and reproducibility.
takes raw Diffusion-weighted and T1-weighted images as input and outputs
diffusion metrics and a whole brain tractogram.

To pay attention
================

-  The singularity.conf file is not necessary for newer versions of
   singularity: ``nextflow -c singularity.conf run <COMMAND>``

-  When using tractoflow use ``--input <DATASET_ROOT_FOLDER>`` instead
   of ``--root <DATASET_ROOT_FOLDER>``

ROI segmentation
================

Establishing correspondences across brains for the purposes of
comparison and group analysis is almost universally done by registering
images to one another either directly or via a template.

1. De-skulling aids volume registration methods.
2. Custom-made optimal average templates improve registration over
   direct pairwise registration.

RecobundleX
===========

Automatic bundle segmentation was done using rbx_flow, which is a fully
automated and reproducible pipeline based on the dipy tool Recobundle.
### To pay attention \* If you download Ants it can be helpful to change
the command ``make -j 4`` to ``make -j 1`` to not overflow RAM. Install
with the following `instructions on
GitHub <https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS>`__
or on `NeuroDebian <https://neuro.debian.net/pkgs/ants.html>`__ (this
did not work for me because the package is not available for my
distribution – Ubuntu 20.04 Focal Fossa) \*
``scil_recognize_multi_bundles.py`` use –outdir and not –output

Nextflow
--------

RecobundleX can be run using a nextflow pipeline called ``rbx_flow``
that can be found `here <https://github.com/scilus/rbx_flow>`__. It is
important to note that it must be run with singularity DSL1 because
:literal:`Operator `into` has been deprecated -- it's not available in DSL2 syntax`.
To use DSL1 run command ``export NXF_DEFAULT_DSL=1``

Our pipeline (bundle_segmentation original)
===========================================

Nextflow pipeline: 1. Register Atlas to reference image: brainnetome
atlas is sent to subject space. To compute transformation from atlas
space to subject space we register ‘FSL_HCP1065_FA_1mm’ to the subject’s
native map. We then apply transformation (affine + warp) on Brainnetome
atlas with generic_label interpolation. 2. Create masks of mPFC and NA:
these are the ROIs for filtering the tractogram based on the atlas 3.
Filter tractogram: remove fibers that do not join both ROIs and
remaining outliers 4. Compute centroid: important step for tractometry
