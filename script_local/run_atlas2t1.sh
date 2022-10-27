#!/bin/bash


subject_ID='DCL_PT_PL007_V1'
template='/home/pabaua/Downloads/mni_icbm152_nlin_asym_09a_nifti/mni_icbm152_nlin_asym_09a/mni_icbm152_t1_tal_nlin_asym_09a.nii'
t1_image='/home/pabaua/dev_tpil/results/results_tracto/DCL_PT_PL007_V1/Bet_T1/DCL_PT_PL007_V1__t1_bet.nii.gz'
atlas='/home/pabaua/dev_scil/atlas/BN_Atlas_246_1mm.nii.gz'
fa_image='/home/pabaua/dev_tpil/results/results_tracto/DCL_PT_PL007_V1/DTI_Metrics/DCL_PT_PL007_V1__fa.nii.gz'
output='/home/pabaua/dev_tpil/data/connectflow_data/DCL_PT_PL007_V1'

cd ${output}
cp ${t1_image} DCL_PT_PL007_V1__t1.nii.gz
rm ${output}/DCL_PT_PL007_V1__t1_bet.nii.gz

t1_image=${output}/DCL_PT_PL007_V1__t1.nii.gz

antsRegistrationSyNQuick.sh -d 3 -f ${template} -m ${atlas} -t s -o ${subject_ID}_atlas2template__
antsRegistrationSyNQuick.sh -d 3 -f ${template} -m ${t1_image} -t s -o ${subject_ID}_t12template__


affine=${subject_ID}_atlas2template__0GenericAffine.mat
warp=${subject_ID}_atlas2template__1Warp.nii.gz
antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${template} -o ${subject_ID}__labels.nii.gz -n genericLabel -u int

#antsRegistrationSyNQuick.sh -d 3 -f ${fa_image} -m ${t1_image} -t s -o ${subject_ID}_t12fa__
