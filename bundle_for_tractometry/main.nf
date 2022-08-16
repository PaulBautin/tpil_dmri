#!/usr/bin/env nextflow

if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()

    bindings = ["atlas_config":"$params.atlas",
                "run_average_bundles":"$params.run_average_bundles",
                "multi_parameters":"$params.multi_parameters",
                "minimal_vote_ratio":"$params.minimal_vote_ratio",
                "wb_clustering_thr":"$params.wb_clustering_thr",
                "seeds":"$params.seeds",
                "outlier_alpha":"$params.outlier_alpha",
                "register_processes":"$params.register_processes",
                "rbx_processes":"$params.rbx_processes",
                "cpu_count":"$cpu_count"]

    engine = new groovy.text.SimpleTemplateEngine()
    template = engine.createTemplate(usage.text).make(bindings)
    print template.toString()
    return
}

log.info "TPIL New Bundle pipeline"
log.info "=========================="
log.info ""
log.info "Start time: $workflow.start"
log.info ""

log.debug "[Command-line]"
log.debug "$workflow.commandLine"
log.debug ""

log.info "[Git Info]"
log.info "$workflow.repository - $workflow.revision [$workflow.commitId]"
log.info ""

log.info "Options"
log.info "======="
log.info ""
log.info "[Atlas]"
log.info "Atlas: $params.atlas"
log.info "Atlas Anat: $params.atlas_anat"
log.info "Atlas Directory: $params.atlas_directory"
log.info "Atlas Centroids: $params.atlas_centroids"
log.info ""
log.info "[Recobundles options]"
log.info "Multi-Parameters Executions: $params.multi_parameters"
log.info "Minimal Vote Percentage: $params.minimal_vote_ratio"
log.info "Whole Brain Clustering Threshold: $params.wb_clustering_thr"
log.info "Random Seeds: $params.seeds"
log.info "Outlier Removal Alpha: $params.outlier_alpha"
log.info ""
log.info ""

log.info "Input: $params.input"

if (!(params.atlas)) {
    error "You must specify all atlas related input. --atlas, "
}

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}

process Register_Atlas_to_Ref {
    cpus params.register_processes
    memory '2 GB'

    input:
    tuple val(sid), file(native_anat), file(atlas)

    output:
    tuple val(sid), file("${sid}__output0GenericAffine.mat"), file("${sid}__output1Warp.nii.gz")

    script:
    """
    echo ${sid}
    echo ${native_anat}
    echo ${atlas}
    antsRegistrationSyNQuick.sh -d 3 -f ${native_anat} -m ${atlas} -t s -o ${sid}__output
    # cp ${native_anat} ${sid}__native_anat.nii.gz
    """
}

process Apply_transform {

    input:
    tuple val(sid), file(native_anat), file(atlas), file(affine), file(warp)

    output:
    tuple val(sid), file("${sid}__atlas_transformed_int.nii.gz")

    script:
    """
    antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${native_anat} -o ${sid}__atlas_transformed.nii.gz -n genericLabel -v 1 --float 0
    scil_image_math.py convert ${sid}__atlas_transformed.nii.gz ${sid}__atlas_transformed_int.nii.gz --data_type int16
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
    tuple val(sid), file("${sid}_trk_cleaned.trk")

    script:
    """
    scil_filter_tractogram.py ${tractogram} ${sid}_trk_filtered.trk \
    --drawn_roi ${mask_mPFC} either_end include \
    --drawn_roi ${mask_NAC} either_end include
    scil_outlier_rejection.py ${sid}_trk_filtered.trk ${sid}_trk_cleaned.trk --alpha 0.5
    """
}


process Compute_Centroid {

    input:
    tuple val(sid), file(bundle)

    output:
    tuple val(sid), file("${sid}_centroid.trk")

    script:
    """
    scil_compute_centroid.py ${bundle} ${sid}_centroid.trk --nb_points 50
    """
}


workflow {
    root = file(params.input)
    /* Watch out, files are ordered alphabetically in channel */
    atlas = Channel.fromPath("$params.atlas")
    tractogram_for_filtering = Channel.fromPath("$root/**/tractogram/*__tractogram.trk").map{[it.parent.parent.name, it]}
    ref_images = Channel.fromPath("$root/**/ref_image/*__ref_image.nii.gz").map{[it.parent.parent.name, it]}

    main:
    ref_images.combine(atlas).set{data_registration}
    Register_Atlas_to_Ref(data_registration)

    data_registration.join(Register_Atlas_to_Ref.out, by:0).set{data_for_transfo}
    data_for_transfo.view()
    Apply_transform(data_for_transfo)

    Create_mask(Apply_transform.out)

    tractogram_for_filtering.combine(Create_mask.out, by:0).set{data_for_filtering}
    data_for_filtering.view()
    Filter_tractogram(data_for_filtering)

    Compute_Centroid(Filter_tractogram.out)


}
