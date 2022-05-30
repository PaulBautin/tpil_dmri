# tpil_dmri
The starting point and aim of this project is to use diffusion MRI to improve our understanding of chronic pain and its treatment.

## Dataset
Raw DWI images are acquired with the same protocol to offer a homogeneous dataset:  b-values of 300, 1000 and 2000 with a resolution of $2\times 2\times 2 mm^3$.

## dMRI QC
Using the *dmri_qc* nextflow pipeline it is possible to check DWIs and the processed DWIs.

**Urls:**
* https://github.com/scilus/dmriqcpy
* https://github.com/scilus/dmriqc_flow

**Example command:**

`nextflow run dmriqc-flow-0.1.2/main.nf -profile input_qc --root input/ -with-singularity singularity_dmriqc_0.1.2.img -resume --raw_dwi_nb_threads 10`

## TractoFlow
Using the *TractFlow* nextflow pipeline it is possible to compute necessary derivatives: DTI metrics, fODF metrics. The script used to run is `run_tractoflow`

**Urls:**
* https://github.com/scilus/tractoflow/

**Example command:**

`nextflow run main.nf --input <DATASET_ROOT_FOLDER> --dti_shells "0 300 1000" --fodf_shells "0 1000 1200" -with-singularity`


## RecobundleX
Using the *rbx_flow* nextflow pipeline it is possible to extract white matter fiber bundles of interest. The script used to run is `run_rbxflow`

**Urls:**
* https://github.com/scilus/tractoflow/

**Example command:**

`nextflow run main.nf -resume -with-singularity scilus-1.2.0_rbxflow-1.1.0.img --input input/ --atlas_config code/rbx-atlas/config.json --atlas_anat code/rbx-atlas/mni_masked.nii.gz --atlas_directory code/rbx-atlas/atlas/`
