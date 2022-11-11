#!/usr/bin/env nextflow

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
log.info "Atlas: $params.atlas"
log.info "Template: $params.template"
log.info ""
log.info "[Filtering options]"
log.info "Outlier Removal Alpha: $params.outlier_alpha"
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
    tuple val(sid), file("${sid}__mask_mPFC27.nii.gz"), file("${sid}__mask_mPFC45.nii.gz"), file("${sid}__mask_mPFC47.nii.gz"), file("${sid}__mask_NAC.nii.gz")

    script:
    """
    #!/usr/bin/env python3
    import nibabel as nib
    atlas = nib.load("$atlas")
    data_atlas = atlas.get_fdata()

    # Create mask mPFC
    mask_mPFC27 = (data_atlas == 27)
    mPFC_27 = nib.Nifti1Image(mask_mPFC27.astype(int), atlas.affine)
    nib.save(mPFC_27, '${sid}__mask_mPFC27.nii.gz')

    # Create mask mPFC
    mask_mPFC45 = (data_atlas == 45)
    mPFC_45 = nib.Nifti1Image(mask_mPFC45.astype(int), atlas.affine)
    nib.save(mPFC_45, '${sid}__mask_mPFC45.nii.gz')

    # Create mask mPFC
    mask_mPFC47 = (data_atlas == 47)
    mPFC_47 = nib.Nifti1Image(mask_mPFC47.astype(int), atlas.affine)
    nib.save(mPFC_47, '${sid}__mask_mPFC47.nii.gz')

    # Create mask NAC
    mask_NAC = (data_atlas == 219) | (data_atlas == 223)
    NAC = nib.Nifti1Image(mask_NAC.astype(int), atlas.affine)
    nib.save(NAC, '${sid}__mask_NAC.nii.gz')
    """
}

process Filter_tractogram {
    input:
    tuple val(sid), file(tractogram), file(mask_mPFC27), file(mask_mPFC45), file(mask_mPFC47), file(mask_NAC)

    output:
    tuple val(sid), file("${sid}__NAC_mPFC_L_cleaned_27.trk"), file("${sid}__NAC_mPFC_L_cleaned_45.trk"), file("${sid}__NAC_mPFC_L_cleaned_47.trk")

    script:
    """
    scil_filter_tractogram.py ${tractogram} ${sid}_trk_filtered_27.trk \
    --drawn_roi ${mask_mPFC27} either_end include \
    --drawn_roi ${mask_NAC} either_end include
    scil_outlier_rejection.py ${sid}_trk_filtered_27.trk ${sid}__NAC_mPFC_L_cleaned_27.trk --alpha 0.6

    scil_filter_tractogram.py ${tractogram} ${sid}_trk_filtered_45.trk \
    --drawn_roi ${mask_mPFC45} either_end include \
    --drawn_roi ${mask_NAC} either_end include
    scil_outlier_rejection.py ${sid}_trk_filtered_45.trk ${sid}__NAC_mPFC_L_cleaned_45.trk --alpha 0.6

    scil_filter_tractogram.py ${tractogram} ${sid}_trk_filtered_47.trk \
    --drawn_roi ${mask_mPFC47} either_end include \
    --drawn_roi ${mask_NAC} either_end include
    scil_outlier_rejection.py ${sid}_trk_filtered_47.trk ${sid}__NAC_mPFC_L_cleaned_47.trk --alpha 0.6
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

process bundle_QC {
    input:
    tuple val(sid), file(ref_image), file(bundle27), file(bundle45), file(bundle47)

    output:
    tuple val(sid), file("${sid}_mosaic.png")

    script:
    """
    scil_visualize_bundles_mosaic.py $ref_image $bundle27 $bundle45 $bundle47 ${sid}_mosaic.png
    """
}


workflow {
    /* Input files to fetch */
    root = file(params.input)
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    tractogram_for_filtering = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.name, it]}
    ref_images = Channel.fromPath("$root/*/*__ref_image.nii.gz").map{[it.parent.name, it]}

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    ref_images.combine(template).set{data_registration}
    Register_Template_to_Ref(data_registration)

    /* Appy registration transformation to atlas  */
    ref_images.combine(atlas).join(Register_Template_to_Ref.out, by:0).set{data_transfo}
    data_transfo.view()
    Apply_transform(data_transfo)

    /* Create ROI masks (based on atlas) for filtering tractogram  */
    Create_mask(Apply_transform.out)

    /* Filter tractogram based on ROI masks  */
    tractogram_for_filtering.join(Create_mask.out, by:0).set{data_for_filtering}
    Filter_tractogram(data_for_filtering)

    /* Compute segmented bundle centroid  */
    Compute_Centroid(Filter_tractogram.out)

    /* Quality control of bundles  */
    ref_images.join(Filter_tractogram.out, by:0).set{bundles_for_qc}
    bundles_for_qc.view()
    bundle_QC(bundles_for_qc)
}
