#!/bin/bash

# This would run fMRIprep with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects);
#   - with-singularity: container image fMRIprep 23.2.0


#SBATCH --nodes=1              # --> Generally depends on your nb of subjects.
                               # See the comment for the cpus-per-task. One general rule could be
                               # that if you have more subjects than cores/cpus (ex, if you process 38
                               # subjects on 32 cpus-per-task), you could ask for one more node.
#SBATCH --cpus-per-task=32     # --> You can see here the choices. For beluga, you can choose 32, 40 or 64.
                               # https://docs.computecanada.ca/wiki/B%C3%A9luga/en#Node_Characteristics
#SBATCH --mem=0                # --> 0 means you take all the memory of the node. If you think you will need
                               # all the node, you can keep 0.
#SBATCH --time=6:00:00

#SBATCH --mail-user=paul.bautin@polymtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


module load StdEnv/2020 apptainer/1.1.8

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/fmriprep-23.2.0.simg' # or .sif
my_input='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset'
my_output='/home/pabaua/scratch/tpil_dev/results/all/2024-02-02_fmriprep_v1/'
my_work='/home/pabaua/scratch/tpil_dev/results/all/2024-02-02_fmriprep_v1/work/'
fs_dir='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/freesurfer_v2'
bids_filter='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/tpil_dmri/script_local/fmriprep_bids_filter_v2.json'

my_licence_fs='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/license.txt'


for subject in ${my_input}/sub-*; do
export APPTAINERENV_FS_LICENSE=$my_licence_fs
apptainer exec --cleanenv -B /project:/project -B /scratch:/scratch $my_singularity_img env | grep FS_LICENSE
apptainer run --cleanenv -B /project:/project -B /scratch:/scratch $my_singularity_img $my_input $my_output participant -w $my_work --output-spaces T1w --participant-label ${subject##*sub-} --cifti-output 91k --bids-filter-file $bids_filter --fs-subjects-dir $fs_dir; 
done