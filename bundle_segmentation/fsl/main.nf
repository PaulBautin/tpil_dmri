#!/usr/bin/env nextflow
nextflow.enable.dsl=2

if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()

    bindings = ["outlier_alpha":"$params.outlier_alpha",
                "cpu_count":"$cpu_count"]

    engine = new groovy.text.SimpleTemplateEngine()
    template = engine.createTemplate(usage.text).make(bindings)
    print template.toString()
    return
}

log.info ""
log.info "TPIL Bundle Segmentation Pipeline"
log.info "=================================="
log.info "Start time: $workflow.start"
log.info ""
log.info "[Input info]"
log.info "Input Folder: $params.input"
log.info ""
log.info "[Filtering options]"
log.info "Outlier Removal Alpha: $params.outlier_alpha"
log.info "Source ROI: $params.source_roi"
log.info "Target ROI: $params.target_roi"
log.info ""

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}


process Create_sub_mask {

    input:
    tuple val(sid), path(t1_image), path(affine), path(warp), file(t1_diff)

    output:
    tuple val(sid), file("${sid}__first_atlas_transformed.nii.gz")

    script:
    """
    run_first_all -i ${t1_image} -o ${sid} -b -v
    antsApplyTransforms -d 3 -i ${sid}_all_fast_firstseg.nii.gz -t ${affine} -t ${warp} -r ${t1_diff} -o ${sid}__first_atlas_transformed.nii.gz -n genericLabel -u int
    """
}

process freesurfer_to_subject {
    memory_limit='6 GB'
    cpus=4

    input:
    tuple val(sid), path(fs_label), path(fs_sphere), path(brain_template), file(fs_atlas), file(t1_diff)

    output:
    tuple val(sid), file("lh.BN_Atlas.annot"), file("lh.BN_Atlas.nii.gz"), file("rh.BN_Atlas.annot"), file("rh.BN_Atlas.nii.gz")

    script:
    """
    tkregister2 --mov $SUBJECTS_DIR/${sid}/mri/brain.mgz --noedit --s ${sid} --regheader --reg register.dat
    mris_ca_label -l $SUBJECTS_DIR/${sid}/label/lh.cortex.label ${sid} lh sphere.reg $SUBJECTS_DIR/lh.BN_Atlas.gcs lh.BN_Atlas.annot -t $SUBJECTS_DIR/BN_Atlas_210_LUT.txt
    cp $SUBJECTS_DIR/${sid}/label/lh.BN_Atlas.annot .
    mris_ca_label -l $SUBJECTS_DIR/${sid}/label/rh.cortex.label ${sid} rh sphere.reg $SUBJECTS_DIR/rh.BN_Atlas.gcs rh.BN_Atlas.annot -t $SUBJECTS_DIR/BN_Atlas_210_LUT.txt
    cp $SUBJECTS_DIR/${sid}/label/rh.BN_Atlas.annot .
    mri_label2vol --subject ${sid} --hemi lh --annot BN_Atlas --o $SUBJECTS_DIR/${sid}/mri/lh.BN_Atlas.nii.gz --temp $SUBJECTS_DIR/${sid}/mri/brain.mgz --reg register.dat --proj frac 0 1 0.01
    cp $SUBJECTS_DIR/${sid}/mri/lh.BN_Atlas.nii.gz .
    mri_label2vol --subject ${sid} --hemi rh --annot BN_Atlas --o $SUBJECTS_DIR/${sid}/mri/rh.BN_Atlas.nii.gz --temp $SUBJECTS_DIR/${sid}/mri/brain.mgz --reg register.dat --proj frac 0 1 0.01
    cp $SUBJECTS_DIR/${sid}/mri/rh.BN_Atlas.nii.gz .
    """
}

process freesurfer_transform {
    memory_limit='6 GB'
    cpus=4

    input:
    tuple val(sid), path(fs_seg_lh), path(fs_seg_rh), file(t1_diff)

    output:
    tuple val(sid), file("${sid}__atlas_transformed.nii.gz"), file("out_labels_2.nii.gz")

    script:
    """
    mri_convert $SUBJECTS_DIR/${sid}/mri/brainmask.mgz mask_brain.nii.gz
    scil_image_math.py lower_threshold mask_brain.nii.gz 1 mask_brain_bin.nii.gz
    scil_combine_labels.py out_labels.nii.gz --volume_ids ${fs_seg_lh} all --volume_ids ${fs_seg_rh} all
    scil_dilate_labels.py out_labels.nii.gz out_labels_2.nii.gz --distance 1.5 --mask mask_brain_bin.nii.gz
    antsRegistrationSyNQuick.sh -d 3 -f ${t1_diff} -m $SUBJECTS_DIR/${sid}/mri/brain.mgz -t s -o ${sid}__output
    antsApplyTransforms -d 3 -i out_labels_2.nii.gz -t ${sid}__output1Warp.nii.gz -t ${sid}__output0GenericAffine.mat -r ${t1_diff} -o ${sid}__atlas_transformed.nii.gz -n genericLabel -u int
    """
}

process Create_mask {
    input:
    tuple val(sid), path(fsl_atlas), path(fs_atlas)

    output:
    tuple val(sid), file("${sid}__mask_source_*.nii.gz"), emit: masks_source
    tuple val(sid), file("${sid}__mask_target_*.nii.gz"), emit: masks_target

    script:
    """
    #!/usr/bin/env python3
    import nibabel as nib
    import numpy as np
    fsl_atlas_data = nib.load("$fsl_atlas").get_fdata() + 210
    fs_atlas_data = nib.load("$fs_atlas").get_fdata()

    # Create masks sources
    for s in $params.source_roi:
        mask = (fsl_atlas_data == s + 210)
        mask_img = nib.Nifti1Image(mask.astype(int), nib.load("$fsl_atlas").affine, dtype=np.int16)
        nib.save(mask_img, '${sid}__mask_target_'+str(s)+'.nii.gz')

    # Create masks target
    for t in $params.target_roi:
        mask = (fs_atlas_data == t)
        mask_img = nib.Nifti1Image(mask.astype(int), nib.load("$fs_atlas").affine, dtype=np.int16)
        nib.save(mask_img, '${sid}__mask_source_'+str(t)+'.nii.gz')
    """
}

process Clean_Bundles {
    memory_limit='8 GB'

    input:
    tuple val(sid), file(tractogram), file(mask_source), file(mask_target)

    output:

    tuple val(sid), file("${sid}__${source_nb}_${target_nb}_L_cleaned.trk"), emit: cleaned_bundle
    tuple val(sid), file("${sid}__${source_nb}_${target_nb}_L_filtered.trk"), emit: filtered_bundle

    script:
    source_nb = mask_source.name.split('_source_')[1].split('.nii')[0]
    target_nb = mask_target.name.split('_target_')[1].split('.nii')[0]
    """
    scil_filter_tractogram.py ${tractogram} ${sid}__${source_nb}_${target_nb}_L_filtered.trk --drawn_roi ${mask_target} either_end include --drawn_roi ${mask_source} either_end include
    scil_outlier_rejection.py ${sid}__${source_nb}_${target_nb}_L_filtered.trk ${sid}__${source_nb}_${target_nb}_L_cleaned.trk --alpha 0.6
    """
}

process Register_Anat {
    input:
    tuple val(sid), file(native_anat), file(template)

    output:
    tuple val(sid), file("${sid}__output0GenericAffine.mat"), file("${sid}__output1Warp.nii.gz"), emit: transformations
    tuple val(sid), file("${sid}__native_anat.nii.gz"), emit: native_anat

    script:
    """
    antsRegistrationSyNQuick.sh -d 3 -f ${native_anat} -m ${template} -t s -o ${sid}__output
    cp ${native_anat} ${sid}__native_anat.nii.gz
    """
}


process Register_Bundle {
    input:
    tuple val(sid), file(bundle), file(affine), file(warp), file(template)

    output:
    tuple val(sid), file("${bname}_mni.trk")

    script:
    bname = bundle.name.split('.trk')[0]
    """
    scil_apply_transform_to_tractogram.py $bundle $template $affine ${bname}_mni.trk --in_deformation $warp --reverse_operation
    """
}

process Bundle_Pairwise_Comparaison_Inter_Subject {
    publishDir = {"./results_bundle/$task.process/$b_name"}
    input:
    tuple val(b_name), file(bundles)

    output:
    tuple val(b_name), file("${b_name}.json")

    script:
    """
    scil_evaluate_bundles_pairwise_agreement_measures.py $bundles ${b_name}.json
    """
}

process Bundle_Pairwise_Comparaison_Intra_Subject {
    publishDir = {"./results_bundle/$task.process/$sid"}
    input:
    tuple val(sid), val(b_names), file(bundles)

    output:
    tuple val(sid), file("${b_names}_Pairwise_Comparaison.json")

    script:
    """
    scil_evaluate_bundles_pairwise_agreement_measures.py $bundles ${b_names}_Pairwise_Comparaison.json
    """
}


process Compute_Centroid {
    input:
    tuple val(sid), file(bundle)

    output:
    tuple val(sid), file("${sid}__NAC_mPFC_L_centroid.trk")

    script:
    """
    scil_compute_centroid.py ${bundle} ${sid}__NAC_mPFC_L_centroid.trk --nb_points 50
    """
}

process bundle_QC_screenshot {
    input:
    tuple val(sid), file(bundle), file(ref_image)

    output:
    tuple val(sid), file("${sid}__${bname}.png")

    script:
    bname = bundle.name.split("__")[1].split('_L_')[0]
    """
    scil_visualize_bundles_mosaic.py $ref_image $bundle ${sid}__${bname}.png -f --light_screenshot --no_information
    """
}


workflow {
    /* Input files to fetch */
    root = file(params.input)
    root_fs = file(params.input_fs)
    template = Channel.fromPath("$params.template")
    freesurfer_atlas_lh = Channel.fromPath("/home/pabaua/dev_tpil/data/Freesurfer/lh.BN_Atlas.gcs")
    freesurfer_label_lh = Channel.fromPath("$root_fs/**/lh.cortex.label").map{[it.parent.parent.name, it]}
    freesurfer_sphere_lh = Channel.fromPath("$root_fs/**/lh.sphere.reg").map{[it.parent.parent.name, it]}
    freesurfer_brain = Channel.fromPath("$root_fs/**/brain.mgz").map{[it.parent.parent.name, it]}
    tractogram_for_filtering = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.parent.name, it]}
    t1_images = Channel.fromPath("$root/*/Crop_T1/*__t1_bet_cropped.nii.gz").map{[it.parent.parent.name, it]}
    t1_images_diff = Channel.fromPath("$root/*/Register_T1/*__t1_warped.nii.gz").map{[it.parent.parent.name, it]}
    t1_to_diff_affine = Channel.fromPath("$root/*/Register_T1/*__output0GenericAffine.mat").map{[it.parent.parent.name, it]}
    t1_to_diff_warp = Channel.fromPath("$root/*/Register_T1/*__output1Warp.nii.gz").map{[it.parent.parent.name, it]}
    tractogram_for_filtering = Channel.fromPath("$root/*/Local_Tracking/*_local_tracking_prob_fa_seeding_fa_mask_seed_0.trk").map{[it.parent.parent.name, it]}

    main:
    /* Create ROI masks (based on atlas) for filtering tractogram  */
    t1_images.combine(t1_to_diff_affine, by:0).combine(t1_to_diff_warp, by:0).combine(t1_images_diff, by:0).set{first_transform}
    Create_sub_mask(first_transform)
    freesurfer_label_lh.combine(freesurfer_sphere_lh, by:0).combine(freesurfer_brain, by:0).combine(freesurfer_atlas_lh).combine(t1_images_diff, by:0).set{freesurfer_pr}
    freesurfer_to_subject(freesurfer_pr)
    freesurfer_to_subject.out.map{[it[0], it[2], it[4]]}.combine(t1_images_diff, by:0).set{freesurfer_tr}
    freesurfer_tr.view()
    freesurfer_transform(freesurfer_tr)

    /* Create ROI masks (based on atlas) for filtering tractogram  */
    Create_sub_mask.out.combine(freesurfer_transform.out, by:0).set{data_for_creating_mask}
    Create_mask(data_for_creating_mask)

    /* Filter tractogram based on ROI masks  */
    Create_mask.out.masks_source.view()
    tractogram_for_filtering.combine(Create_mask.out.masks_source, by:0).combine(Create_mask.out.masks_target, by:0).set{data_for_filtering}
    data_for_filtering.view()
    Clean_Bundles(data_for_filtering)

    t1_images_diff.combine(template).set{data_for_anat_registration}
    Register_Anat(data_for_anat_registration)

    Clean_Bundles.out.cleaned_bundle.combine(Register_Anat.out.transformations, by:0).combine(template).set{bundle_registration}
    Register_Bundle(bundle_registration)

    Register_Bundle.out.map{[it[1].name.split('_ses-')[1].split('_L')[0], it[1]]}.groupTuple(by:0).set{bundle_comparaison_inter}
    Bundle_Pairwise_Comparaison_Inter_Subject(bundle_comparaison_inter)

    Register_Bundle.out.map{[it[0].split('_ses')[0], it[1].name.split('__')[1].split('_L_')[0], it[1]]}.groupTuple(by:[0,1]).set{bundle_comparaison_intra}
    Bundle_Pairwise_Comparaison_Intra_Subject(bundle_comparaison_intra)

    Clean_Bundles.out.cleaned_bundle.combine(t1_images_diff, by:0).set{bundles_for_screenshot}
    bundle_QC_screenshot(bundles_for_screenshot)
}

