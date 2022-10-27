#!/usr/bin/env nextflow

if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()

    bindings = ["outlier_alpha":"$params.outlier_threshold",
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
log.info "Atlas: $params.atlas"
log.info "Template: $params.template"
log.info ""
log.info "[Filtering options]"
log.info "Outlier Removal Alpha: $params.outlier_threshold"
log.info ""

if (!(params.atlas) | !(params.template)) {
    error "You must specify an atlas and a template with command line flags. --atlas and --template, "
}

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}

process Register_Template_to_Ref {
    input:
    tuple val(sid), file(native_anat), file(template)

    output:
    tuple val(sid), file("${sid}__output0GenericAffine.mat"), file("${sid}__output1Warp.nii.gz")

    script:
    """
    antsRegistrationSyNQuick.sh -d 3 -f ${native_anat} -m ${template} -t s -o ${sid}__output
    """
}

process Apply_transform {
    input:
    tuple val(sid), file(native_anat), file(atlas), file(affine), file(warp)

    output:
    tuple val(sid), file("${sid}__atlas_transformed.nii.gz")

    script:
    """
    antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${native_anat} -o ${sid}__atlas_transformed.nii.gz -n genericLabel -u int
    """
}

process Create_mask {
    input:
    tuple val(sid), path(atlas)

    output:
    tuple val(sid), file("${sid}__mask_mPFC.nii.gz"), file("${sid}__mask_NAC.nii.gz")

    script:
    """
    #!/usr/bin/env python3
    import nibabel as nib
    atlas = nib.load("$atlas")
    data_atlas = atlas.get_fdata()

    # Create mask mPFC
    mask_mPFC = (data_atlas == 27) | (data_atlas == 45) | (data_atlas == 47)
    mPFC = nib.Nifti1Image(mask_mPFC.astype(int), atlas.affine)
    nib.save(mPFC, '${sid}__mask_mPFC.nii.gz')

    # Create mask NAC
    mask_NAC = (data_atlas == 219) | (data_atlas == 223)
    NAC = nib.Nifti1Image(mask_NAC.astype(int), atlas.affine)
    nib.save(NAC, '${sid}__mask_NAC.nii.gz')
    """
}

process Filter_tractogram {
    input:
    tuple val(sid), file(tractogram), file(mask_mPFC), file(mask_NAC)

    output:
    tuple val(sid), file("${sid}__NAC_mPFC_L_cleaned.trk")

    script:
    """
    scil_filter_tractogram.py ${tractogram} ${sid}_trk_filtered.trk \
    --drawn_roi ${mask_mPFC} either_end include \
    --drawn_roi ${mask_NAC} either_end include
    scil_outlier_rejection.py ${sid}_trk_filtered.trk ${sid}__NAC_mPFC_L_cleaned.trk --alpha 0.6
    """
}

process Decompose_Connectivity {
    input:
    tuple val(sid), file(tractogram), file(atlas)

    output:
    tuple val(sid), file("${sid}__decompose.h5")

    script:
    no_pruning_arg = ""
    if (params.no_pruning) {
        no_pruning_arg = "--no_pruning"
    }
    no_remove_loops_arg = ""
    if (params.no_remove_loops) {
        no_remove_loops_arg = "--no_remove_loops"
    }
    no_remove_outliers_arg = ""
    if (params.no_pruning) {
        no_remove_outliers_arg = "--no_pruning"
    }
    no_remove_outliers_arg = ""
    if (params.no_remove_outliers) {
        no_remove_outliers_arg = "--no_remove_outliers"
    }
    """
    if [ `echo $tractogram | wc -w` -gt 1 ]; then
        scil_streamlines_math.py lazy_concatenate $tractogram tracking_concat.trk
    else
        mv $tractogram tracking_concat.trk
    fi

    scil_decompose_connectivity.py tracking_concat.trk $atlas "${sid}__decompose.h5" --no_remove_curv_dev \
        $no_pruning_arg $no_remove_loops_arg $no_remove_outliers_arg --min_length $params.min_length \
        --max_length $params.max_length --loop_max_angle $params.loop_max_angle \
        --outlier_threshold $params.outlier_threshold
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


workflow {
    /* Input files to fetch */
    root = file(params.input)
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    tractogram = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.name, it]}
    ref_images = Channel.fromPath("$root/*/*__ref_image.nii.gz").map{[it.parent.name, it]}

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    data_registration = ref_images.combine(template)
    Register_Template_to_Ref(data_registration)

    /* Appy registration transformation to atlas  */
    data_transfo = ref_images.combine(atlas).join(Register_Template_to_Ref.out, by:0)
    data_transfo.view()
    Apply_transform(data_transfo)

    /* Create connectivity based on atlas and tractogram  */
    data_connectivity = tractogram.combine(Apply_transform.out, by:0)
    data_connectivity.view()
    Decompose_Connectivity(data_connectivity)

    /* Filter tractogram based on ROI masks
    tractogram_for_filtering.combine(Create_mask.out, by:0).set{data_for_filtering}
    Filter_tractogram(data_for_filtering) */

    /* Compute segmented bundle centroid
    Compute_Centroid(Filter_tractogram.out) */
}
