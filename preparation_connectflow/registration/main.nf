#!/usr/bin/env nextflow

if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()

    bindings = ["cpu_count":"$cpu_count"]

    engine = new groovy.text.SimpleTemplateEngine()
    template = engine.createTemplate(usage.text).make(bindings)
    print template.toString()
    return
}

log.info ""
log.info "TPIL Connectflow Preparation Pipeline"
log.info "=================================="
log.info "Start time: $workflow.start"
log.info ""
log.info "[Input info]"
log.info "Input Folder: $params.input"
log.info "Atlas: $params.atlas"
log.info "Template: $params.template"
log.info ""
log.info "[Filtering options]"
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
    tuple val(sid), file("${sid}__labels.nii.gz")

    script:
    """
    antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${native_anat} -o ${sid}__labels.nii.gz -n genericLabel
    """
}

process Copy_sup_files {
    input:
    tuple val(sid), file(sup_files_1),file(sup_files_2),file(sup_files_3),file(sup_files_4),file(sup_files_5),file(sup_files_6),file(sup_files_7)

    output:
    tuple val(sid), file(sup_files_1),file(sup_files_2),file(sup_files_3),file(sup_files_4),file(sup_files_5),file(sup_files_6),file(sup_files_7)

    script:
    """
    echo $sup_files_1 $sup_files_2 $sup_files_3
    """
}



workflow {
    /* Input files to fetch */
    root = file(params.input)
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    ref_images = Channel.fromPath("$root/*/*/sub*__fa.nii.gz").map{[it.parent.parent.name, it]}
    ref_images.view()
    info = Channel.fromFilePairs("$root/*/*/sub*__{bval_eddy,dwi_eddy_corrected.bvec,dwi_resampled.nii.gz,peaks.nii.gz,pft_tracking_prob_wm_seed_0.trk,t1_warped.nii.gz,fodf.nii.gz}", size: 7, flat: true)

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    ref_images.combine(template).set{data_registration}
    Register_Template_to_Ref(data_registration)

    /* Appy registration transformation to atlas  */
    ref_images.combine(atlas).join(Register_Template_to_Ref.out, by:0).set{data_transfo}
    Apply_transform(data_transfo)

    Copy_sup_files(info)
    Copy_sup_files.out.view()
}