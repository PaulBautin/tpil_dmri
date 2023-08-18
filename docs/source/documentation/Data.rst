Dataset Protocol
================

MRI Protocol
------------

All participants were imaged using a 3.0 T MRI scanner (Philips Ingenia,
Siemens, Canada) across T1-weighted and diffusion weighted contrasts.
T1-weighted images (5 min) were obtained using a MAG prepared (MP)
Gradient Recalled (GR) sequence with repetition time (RT) = 7.9 ms, echo
time (TE) = 3.5 ms, flip angle = 8°, voxel size 1.0 × 1.0 × 1.0 mm3.
Diffusion-weighted images (10 min) were obtained using segmented k-space
(SK) Spin Echo (SE) sequence with repetition time (RT) = 4000 ms, echo
time (TE) = 92 ms, flip angle = 90°, voxel size 2.0 × 2.0 × 2.0 mm3. For
each subject 108 diffusion volumes (7 b = 0 mm2/s, 8 b = 300 mm2/s, 32 b
= 1,000 mm2/s, 60 b = 2,000 mm2/s) were obtained including a b0 with
reverse phase encoding for correcting susceptibility induced
distortions.

Inclusion/Exclusion criteria for CLBP participants
--------------------------------------------------

-  Age ≥ 18 ans
-  Has had CLBP for more than 16 weeks (4 months), with pain (with poor
   irradiation) towards the buttock or lower
-  Visual Analogue Scale score (VAS) ≥ 30 mm (maximum of 100 mm) within
   the last 24 hours
-  Must not use medication. Example: cortisone infiltration in the last
   2 years, chonic opiodes, antidepressants

Dataset info
------------

For this study 25 participants with chronic lower back pain (X males, X
females) and 27 control participants (X males, X females) were recruited
and scanned over a period of 6 months (X to X 2021) in Sherbrooke,
Quebec, Canada.:

-  remove subject 18
-  subjects 37, 39, 8 have only visit v1
-  subject 16 has only visit v1 and v2
-  remove subject 4 v1 because no inverse b0

Dataset manipulation
====================

Brain Imaging Data Structure
----------------------------

A tutorial on converting dicom files to BIDS format can be found at this
`website <https://unfmontreal.github.io/Dcm2Bids/>`__. First create a
conda environment for dcm2bids and install dcm2bids dependencies:

::

   # Activate environment
   conda activate env_dcm2bids
   # Install conda dependencies
   conda install -c conda-forge dcm2bids
   conda install -c conda-forge dcm2niix

Create scaffold for files running command:
``dcm2bids_scaffold -o bids_project``

Enter the new folder and transfer the Dicom files to the *sourcedata*
folder. You are now ready to build a configuration file. An example for
our dataset can be found on this repository.

To run dcm2bids conversion we have created a script (present in this
repository) that renames files before running dcm2bids. Run command:
``bash code/run_dicom_bids.sh``
