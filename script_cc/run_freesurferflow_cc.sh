#!/bin/bash

# This would run Tractoflow with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects), if not in BIDS format use flag --input
#   - with-singularity: container image scilus v1.3.0 (runs: dmriqc_flow, tractoflow, recobundleX, tractometry)
#   - with-report: outputs a processing report when pipeline is finished running
#   - Dti_shells 0 and 1000 (usually <1200), Fodf_shells 0 1000 and 2000 (usually >700, multishell CSD-ms).
#   - profile: bundling, bundling profile will set the seeding strategy to WM as opposed to interface seeding that is usually used for connectomics


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


module load StdEnv/2020 java/14.0.2 nextflow/22.04.3 singularity/3.8


my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/freesurfer_flow/scil_freesurfer_cc.img' # or .img
my_main_nf='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/freesurfer_flow/main.nf'
my_input='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/22-09-21_t1_clbp_freesurfer_output'
atlas_utils='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/freesurfer_flow/FS_BN_GL_SF_utils'


NXF_DEFAULT_DSL=1 nextflow $my_main_nf --root_fs_output $my_input --atlas_utils_folder $atlas_utils \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --compute_lausanne_multiscale false

