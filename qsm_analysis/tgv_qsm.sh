#!/bin/bash

subject=sub-057_ses-v2
data_dir='/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset_v2/sub-057/ses-v2/swi'
output_dir='/home/pabaua/dev_tpil/results/results_qsm'
t1_template='/home/pabaua/dev_tpil/data/Oasis/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz'
t1_prob_map='/home/pabaua/dev_tpil/data/Oasis/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz'

echoTime=(0.005285 0.014285 0.023285 0.032285)

for echoNbr in {1..4}; do
    echo ${data_dir}/${subject}_echo-${echoNbr}_part-mag_GRE.nii.gz
    #bet ${data_dir}/${subject}_echo-${echoNbr}_part-mag_GRE.nii.gz ${output_dir}/${subject}_echo-${echoNbr}_part-mag_GRE_bet2 -m -R -f 0.7
    antsBrainExtraction.sh -d 3 -a ${data_dir}/${subject}_echo-${echoNbr}_part-mag_GRE.nii.gz -e $t1_template -o ${output_dir}/${subject}_echo-${echoNbr}_part-mag_GRE -m $t1_prob_map
done

#sudo docker run -it -v $PWD:/home/pabaua vnmd/tgvqsm_1.0.0:20210317

# for echoNbr in {1..4}; do
#         tgv_qsm \
#         -p ${data_dir}/${subject}_echo-${echoNbr}_part-phase_GRE.nii.gz \
#         -m ${output_dir}/${subject}_echo-${echoNbr}_part-mag_GRE_bet_mask.nii.gz \
#         -f 3 \
#         -e 0 \
#         -t ${echoTime[echoNbr-1]} 
# done