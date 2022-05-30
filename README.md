# tpil_dmri
The starting point and aim of this project is to use diffusion MRI to improve our understanding of chronic pain and its treatment.

## Dataset
Raw DWI images are acquired with the same protocol to offer a homogeneous dataset:  b-values of 300, 1000 and 2000 with a resolution of $2\times 2\times 2 mm^3$.

# Pipelines and scripts
## dMRI QC
Using the *dmri_qc* nextflow pipeline it is possible to check DWIs and the processed DWIs.
<details open>
<summary><b>Resources:</b></summary>
  
  * [Github repository for python](https://github.com/scilus/dmriqcpy)
  * [Github repository for nextflow](https://github.com/scilus/dmriqc_flow)
</details>



<details open>
<summary><b>Example command:</b></summary>

`nextflow run dmriqc-flow-0.1.2/main.nf -profile input_qc --root input/ -with-singularity singularity_dmriqc_0.1.2.img -resume --raw_dwi_nb_threads 10`
</details>

## TractoFlow
Using the *TractFlow* nextflow pipeline it is possible to compute necessary derivatives: DTI metrics, fODF metrics. The script used to run is `run_tractoflow`

<details open>
<summary><b>Resources:</b></summary>

  * [Gihub repository](https://github.com/scilus/tractoflow/)
  * [SCIL TractoFlow documentation](https://scil-documentation.readthedocs.io/en/latest/our_tools/tractoflow.html)
  * [ReadTheDocs TractoFlow documentation](https://tractoflow-documentation.readthedocs.io/en/latest/index.html)
  * `Theaud, G., Houde, J.-C., Boré, A., Rheault, F., Morency, F., Descoteaux, M.,TractoFlow: A robust, efficient and reproducible diffusion MRI pipeline leveraging Nextflow & Singularity, NeuroImage, https://doi.org/10.1016/j.neuroimage.2020.116889.`
</details>

<details open>
<summary><b>Example command:</b></summary>
  
  `nextflow run main.nf --input <DATASET_ROOT_FOLDER> --dti_shells "0 300 1000" --fodf_shells "0 1000 1200" -with-singularity`
</details>

## RecobundleX
Using the *rbx_flow* nextflow pipeline it is possible to extract white matter fiber bundles of interest. The script used to run is `run_rbxflow`

<details open>
<summary><b>Resources:</b></summary>

  * [Github repository](https://github.com/scilus/rbx_flow)
  * [SCIL RecobundleX documentation](https://scil-documentation.readthedocs.io/en/latest/our_tools/recobundles.html)
  * [Example atlases](https://zenodo.org/record/4104300#.YmMEk_PMJaQ)
  * `Rheault, Francois. Analyse et reconstruction de faisceaux de la matière blanche.
page 137-170, (2020), https://savoirs.usherbrooke.ca/handle/11143/17255`
</details>

<details open>
<summary><b>Example command:</b></summary>
  
`nextflow run main.nf -resume -with-singularity scilus-1.2.0_rbxflow-1.1.0.img --input input/ --atlas_config code/rbx-atlas/config.json --atlas_anat code/rbx-atlas/mni_masked.nii.gz --atlas_directory code/rbx-atlas/atlas/`
</details>
