#!/bin/bash

# This would run fMRIprep with the following parameters:
#   - bids: clinical data from TPIL lab (27 CLBP and 25 control subjects);
#   - with-singularity: container image fMRIprep 23.2.0


# container created on CC with command:
# singularity build rapidtide.simg docker://fredericklab/rapidtide:v2.7.8


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

my_singularity_img='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/rapidtide.simg' # or .sif
my_input='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset'
my_output='/home/pabaua/scratch/tpil_dev/results/all/2024-02-02_fmriprep_v1/'
my_work='/home/pabaua/scratch/tpil_dev/results/all/2024-02-02_fmriprep_v1/work/'
fs_dir='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/data/freesurfer_v1'
bids_filter='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_tpil/tpil_dmri/script_local/fmriprep_bids_filter.json'

my_licence_fs='/home/pabaua/projects/def-pascalt-ab/pabaua/dev_scil/containers/license.txt'



singularity run \
    --cleanenv \
    -B INPUTDIRECTORY:/data_in,OUTPUTDIRECTORY:/data_out \
    $my_singularity_img \
    rapidtide \
        /data_in/YOURNIFTIFILE.nii.gz \
        /data_out/outputname \
        --filterband lfo \
        --searchrange -15 15 \
        --passes 3


# Define a function to execute the rapidtide script for each subject ID
run_rapidtide_for_subject() {
  # Assign the subject ID to a variable
  local sid=$1
  local parent_dir=$2

  # Define directories and files based on the subject ID
  local func_dir="${parent_dir}/${sid}" 
  local prepoc_bold="${func_dir}/sub-${sid}_task-rest_space-Native_desc-optcomDenoised_bold.nii.gz"
  local rapidtide_dir="/home/pabaua/dev_tpil/data/fmriprep_derivatives/rapidtide_tedana/${sid}"

  # Create the rapidtide directory for the current subject
  mkdir -p "${rapidtide_dir}"

  # Execute rapidtide
  rapidtide "${prepoc_bold}" "${rapidtide_dir}/${sid}_task-rest_space-Native_desc-optcomDenoised_bold" --filterband lfo --passes 3
}

# Main execution loop: iterate over all folders in the parent directory
parent_dir="/home/pabaua/dev_tpil/results/results_fmriprep/tedana"

for folder in "${parent_dir}"/*; do
  # Extract the subject ID from the folder name (assuming it is the last part of the folder path)
  sid=$(basename "${folder}")
  sid='sub-001'

  # Call the function to execute the rapidtide script for the extracted subject ID
  run_rapidtide_for_subject "${sid}" "${parent_dir}"
done