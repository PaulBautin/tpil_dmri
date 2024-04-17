tractoflow_folder=/home/pabaua/dev_tpil/results/results_tracto/23-09-01_tractoflow_bundling
subject_list=my_subject_list.txt
models_path=YOUR_PATH           # Ex: hcp_models/
model_T1=THE_T1                 # Ex: ${models_path}/mni_masked.nii.gz
model_config=JSON_FILE          # Ex: ${models_path}/config_python.json

# Filtering options (in mm). Change as needed.
minL=20
maxL=200

# RecobundlesX options. Change as needed.
nb_total_executions=9    # len(model_clustering_thr) * len(bundle_pruning_thr) * len(tractogram_clustering_thr) = max total executions (see json).
thresh_dist="10 12"      # Whole brain clustering threshold (in mm) for QuickBundles.
processes=6              # Number of thread used for computation.
seed=0                   # Random number generator initialisation.
minimal_vote=0.5         # Saving streamlines if recognized often enough.


while IFS= read -r subj; do
    echo "Running subject ${subj}"

    # Defining subj folders
    subj_folder=${tractoflow_folder}/${subj}

    # Defining inputs
    subj_trk=${subj_folder}/Tracking/${subj}__local_tracking*.trk
    subj_T1=${subj_folder}/Register_T1/${subj}__t1_warped.nii.gz

    ###
    # Registering model on subject (using ANTS)
    #   -d=image dimension,
    #   -f=fixed image, m=moving image
    #   -t: transformation a = rigid+affine
    #   -n = nb of threads
    # This should create 3 files : model_to_subj_anat0GenericAffine.mat,
    # model_to_subj_anatInverseWarped.nii.gz and  model_to_subj_anatWarped.nii.gz
    ###
    model_to_subj=${rbx_folder}/model_to_subj_anat
    antsRegistrationSyNQuick.sh -d 3 -f ${subj_T1} -m ${model_T1} -t a -n 4 -o ${model_to_subj}

    ###
    # Cleaning tracking file to make sure it is not too big.
    # Recobundles is already long enough :) .
    ###
    subj_filtered_trk=${subj_folder}/Tracking/${subj}__tracking_filteredLength.trk
    scil_filter_streamlines_by_length.py --minL $minL --maxL ${maxL} ${subj_trk} ${subj}_filtered_trk

    ###
    # Scil's Recobundle
    #   processes = nb of threads.
    #   Seed = rnd generator seed.
    #   inverse is to use the inverse affine
    ###
    mkdir ${rbx_folder}/multi_bundles
    scil_recognize_multi_bundles.py ${subj}_filtered_trk ${model_config} ${atlas_dir} ${affine} \
        --out_dir ${rbx_folder}/multi_bundles \
        --processes ${processes} --seeds ${seed}  \
        --minimal_vote_ratio ${minimal_vote} \
        --log_level DEBUG --inverse -f
done < ${subject_list}