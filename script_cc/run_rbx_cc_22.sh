#!/bin/bash

# This would run RecobundleX with the following parameters:
#   - Population average atlas for RecobundlesX. DOI: 10.5281/zenodo.5165374


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


module load StdEnv/2020 java/14.0.2 nextflow/22.04.3 apptainer/1.1

git -C /home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/rbx_flow checkout 1.2.0

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/scilus_1.5.0.sif' # or .img
my_main_nf='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/rbx_flow/main.nf'
my_input='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/23-09-08_rbx_con'
my_atlas_config='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/atlas_v22/config/config_ind.json'
my_atlas_anat='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/atlas_v22/atlas/mni_masked.nii.gz'
my_atlas_dir='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/atlas_v22/atlas/atlas'
my_atlas_centroids='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/atlas_v22/atlas/centroids'


NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume \
    --atlas_config $my_atlas_config --atlas_anat $my_atlas_anat --atlas_directory $my_atlas_dir --atlas_centroids $my_atlas_centroids -profile large_dataset
