# dMRI & Chronic Pain
The starting point and aim of this project is to use diffusion MRI to improve our understanding of chronic pain and its treatment.

## Dataset
Raw DWI images are acquired with the same protocol to offer a homogeneous dataset:  b-values of 300, 1000 and 2000 with a resolution of $2\times 2\times 2 mm^3$.

# Pipelines and scripts
Before running the pipelines and scripts it is required to install [Nextflow](https://www.nextflow.io/), [Singularity](https://sylabs.io/docs/), [Scilpy](https://scil-documentation.readthedocs.io/en/latest/our_tools/scilpy.html), and [Ants](https://scil-documentation.readthedocs.io/en/latest/tools/ants.html). A general installation procedure can be found [here](https://tractoflow-documentation.readthedocs.io/en/latest/installation/requirements.html) and more specific instructions, for each pipeline, can be found in their respective sections.  

If the pipelines are ran on Linux, it is recommended to use Singularity. Pre-built singularity images and container related files for SCILUS flows can be found on the github repository [containers-scilus](https://github.com/scilus/containers-scilus) or on the [SCIL website](http://scil.usherbrooke.ca/en/containers_list/).

## dMRI QC
Using the *dmri_qc* nextflow pipeline it is possible to check DWIs and the processed DWIs. A more in depth description and installation intruction  can be found [here](https://scil-documentation.readthedocs.io/en/latest/our_tools/other_pipelines.html).
<details><summary><b>Resources</b></summary>
  
  * [Github repository for python](https://github.com/scilus/dmriqcpy)
  * [Github repository for nextflow](https://github.com/scilus/dmriqc_flow)
</details>



<details><summary><b>Example command</b></summary>

`nextflow run dmriqc-flow-0.1.2/main.nf -profile input_qc --root input/ -with-singularity singularity_dmriqc_0.1.2.img -resume --raw_dwi_nb_threads 10`
</details>

## TractoFlow
Using the *TractFlow* nextflow pipeline it is possible to compute necessary derivatives: DTI metrics, fODF metrics. The script used to run is `run_tractoflow`

<details><summary><b>Resources</b></summary>

  * [Gihub repository](https://github.com/scilus/tractoflow/)
  * [SCIL TractoFlow documentation](https://scil-documentation.readthedocs.io/en/latest/our_tools/tractoflow.html)
  * [ReadTheDocs TractoFlow documentation](https://tractoflow-documentation.readthedocs.io/en/latest/index.html)
  * `Theaud, G., Houde, J.-C., Boré, A., Rheault, F., Morency, F., Descoteaux, M.,TractoFlow: A robust, efficient and reproducible diffusion MRI pipeline leveraging Nextflow & Singularity, NeuroImage, https://doi.org/10.1016/j.neuroimage.2020.116889.`
</details>

<details><summary><b>Example command</b></summary>
  
  `nextflow run main.nf --input <DATASET_ROOT_FOLDER> --dti_shells "0 300 1000" --fodf_shells "0 1000 1200" -with-singularity`
</details>
  
<details><summary><b>Outputs</b></summary>
  
**DTI metrics**: The Diffusion Tensor Imaging metrics computed are: the axial diffusivity (AD), fractional anisotropy (FA), geodesic anisotropy (GA), mean diffusivity (MD), radial diffusivity (RD), tensor, tensor norm, tensor eigenvalues, tensor eigenvectors, tensor mode, color-FA. Use flag `-dti_shell` to specify the desired shells t compute DTI metrics. Usually it is recommended to compute DTI metrics using b-values under $b = 1200 mm^2/s$.
  
**fODF metrics**: The fiber Orientation Distribution Function metrics computed are: the total and maximum Apparent Fiber Density (AFD), the Number of Fiber Orientation (NuFO) and principal fODFs orientations (up to 5 per voxel). Use flag `–fodf_shells` to specify the desired shells to compute fODF metrics and flag --sh_order to specify the spherical harmonic order (default is 8). Usually it is recommended to compute fODF metrics using b-values above $b = 700 mm^2/s$.
</details>

## RecobundleX
Using the *rbx_flow* nextflow pipeline it is possible to extract white matter fiber bundles of interest. The script used to run is `run_rbxflow`. 

<details><summary><b>Resources</b></summary>

  * [Github repository](https://github.com/scilus/rbx_flow)
  * [SCIL RecobundleX documentation](https://scil-documentation.readthedocs.io/en/latest/our_tools/recobundles.html)
  * [Example atlases](https://zenodo.org/record/4104300#.YmMEk_PMJaQ)
  * `Rheault, Francois. Analyse et reconstruction de faisceaux de la matière blanche.
page 137-170, (2020), https://savoirs.usherbrooke.ca/handle/11143/17255`
</details>

<details><summary><b>Example command</b></summary>
  
`nextflow run main.nf -resume -with-singularity scilus-1.2.0_rbxflow-1.1.0.img --input input/ --atlas_config code/rbx-atlas/config.json --atlas_anat code/rbx-atlas/mni_masked.nii.gz --atlas_directory code/rbx-atlas/atlas/`
</details>
