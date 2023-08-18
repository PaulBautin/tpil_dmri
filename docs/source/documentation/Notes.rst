Questions
---------

dMRI
~~~~

-  The fODF metrics are computed using Single Shell Single Tissue
   Constrained Sperical Deconvolution (SSST-CSD) method. How come i can
   input multiple shells in Tractoflow with flag ``--fodf_shells``?
   Actually the same FRF is used for multiple shells so this method is
   only slightly better than using only one shell. Example 2000
-  Does the manual FRF need to be changed when using different
   fODF_shells? Probably this comes back to using MSMT-CSD
-  Should the Atlas be registered from FA template to FA subject or from
   T1-template to anisotropic power map (APM)? FA is computed in
   TractoFlow so use that one.
-  Is it interesting to change the FRF comparing CLBP patients and
   healthy controls (multiple subject studies)?

Registration
~~~~~~~~~~~~

-  Should ROI segmentation for bundle segmentation be done with
   Freesurfer (surface based parcellation) or ANTs (volume based
   parcellation)?
   `article <https://www.pnas.org/doi/10.1073/pnas.1801582115>`__? Sould
   `SET <https://www.sciencedirect.com/science/article/pii/S1053811917310583>`__
   be used? It is not optimal to use T1_warped in Freesurfer because the
   T1_warped has diffusion deformations and artifacts.
-  NAC sub-divisions do not have the same function in pain encoding.
   Could subdividing be possible to extract only relevant bundle? Yes,
   but is the NAC subdivion to small for relevant tractometry analysis?

Pain
~~~~

-  Does chronic pain develop in children and adolescents (Connection
   between limbic system and PFC is not as well estblished...)
-  Do certain genes correlate with chronic pain (genetic aspect of
   chronic pain..)? Neuromaps again..
-  What is the role of the amygdala in chronic pain -- and other
   important limbic areas?

Segmentation
~~~~~~~~~~~~

-  Use similarity metrics to evaluate bundle segmentation. Already
   implemented for Connectflow.
-  Tractometry based on bundle segmentation and not the atlas
   (clustering quickbundles, atlas based, machine learning)? Would this
   be allright for group comparaisons?
-  Redefine the ROI based on nucleus accumbens and are own similarity
   metrics (same way brainnetome atlas is parcellated)
-  Non-atlas based projection of white matter metrics to the cortex?
   `Atlas-free
   method <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8249904/>`__,
   `connectome
   embedding <https://www.sciencedirect.com/science/article/pii/S1053811921007424?via%3Dihub>`__

Similarity metrics (intra-subject, inter-subject)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  scil_evaluate_bundles_pairwise_agreement_measures.py​
-  scil_evaluate_connectivity_pairwise_agreement_measures.py​
-  Compare methods (Parcellation, aggregation parameters, BST,
   registration for similarity metric computation, COMMIT)

Connectivity:
~~~~~~~~~~~~~

-  try to keep about 30 to 50 % of connections
-  Use PFT for connectflow
-  Add brainstem (and missing ROI) to brainnetome

Parcel-to-parcel:
~~~~~~~~~~~~~~~~~

-  Create average bundle and visualize
-  Compute volume of the mPFC and NAcc segmentation
-  Re-segment bundle in 10 to 20 sections

Ideas
-----

-  Using diffusion MRI for the spinal cord `application to
   pain <https://www.sciencedirect.com/science/article/pii/S1053811920309241?fbclid=IwAR1_ozsTHDwCpYWpga7-50AsQ-Uc3BmBXdFaH5YqnTwK4FFxBZEL_8oiDL0>`__,
   `toolbox <https://www.researchgate.net/publication/316630544_HARDI_dMRI_imaging_of_cervical_spinal_cord>`__
-  NAC parcellation because NAC sub-divisions do not have the same
   function in pain encoding. `shell-core
   dichotomy <https://onlinelibrary.wiley.com/doi/10.1002/hbm.23636>`__
-  Using neuromaps for structural and functional interpretation of brain
   maps. `Neuromaps <https://github.com/netneurolab/neuromaps>`__.
   Relate neurotransmitter receptors and transporters to patterns of
   cortical abnormality
-  Project all streamlines from the limbic system to the cortex, to map
   the structural connectivity differences in chronic pain patients.
   `Surface‐Based Connectivity
   Integration <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8249904/>`__
-  Can structural connectivity predict functional chronic pain
   abnormalities + work from Masha?
   `NBS-predict <https://www.sciencedirect.com/science/article/pii/S1053811921008983#fig0001>`__
-  Look at functional connectivity from NAcc and mPFC.
-  Use the brain state idea to investigate: (i) if there is a different
   fractional occupancy of certain states in chronic pain, (ii) those
   the state of pain reorganize (iii) Network control theory, if the
   pain state reorganizes then is it easier to attain (iv) idea of
   sensory vs cognitive state
-  Using personalized parcellation to go beyond canonical networks and
   look at the reconfiguration of each region (ex. region real-estate).
-  structural-functional coupling and epicenter phenomenon (regions that
   wire together die together).

Bourses:

RBIQ IRSC CNS ​FMSS: mars FRQS: automne FRNRT ou FRQS (Jean François
Delabre)?
