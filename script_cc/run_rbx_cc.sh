#!/bin/bash

# This would run RecobundleX with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


#SBATCH --nodes=1              # --> Generally depends on your nb of subjects.
                               # See the comment for the cpus-per-task. One general rule could be
                               # that if you have more subjects than cores/cpus (ex, if you process 38
                               # subjects on 32 cpus-per-task), you could ask for one more node.
#SBATCH --cpus-per-task=32     # --> You can see here the choices. For beluga, you can choose 32, 40 or 64.
                               # https://docs.computecanada.ca/wiki/B%C3%A9luga/en#Node_Characteristics
#SBATCH --mem=0                # --> 0 means you take all the memory of the node. If you think you will need
                               # all the node, you can keep 0.
#SBATCH --time=24:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


my_singularity_img='/home/pabaua/scratch/scil_dev/containers/scilus_1.3.0.sif' # or .img
my_main_nf='/home/pabaua/scratch/scil_dev/rbx_flow/main.nf'
my_input='/home/pabaua/scratch/tpil_dev/data/dataset_sansan_rbx'
my_atlas_config='/home/pabaua/scratch/scil_dev/atlas/config.json'
my_atlas_anat='/home/pabaua/scratch/scil_dev/atlas/mni_masked.nii.gz'
my_atlas_dir='/home/pabaua/scratch/scil_dev/atlas/atlas'
my_atlas_centroids='/home/pabaua/scratch/scil_dev/atlas/centroids'



NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --atlas_config $my_atlas_config --atlas_anat $my_atlas_anat --atlas_directory $my_atlas_dir --atlas_centroids $my_atlas_centroids
