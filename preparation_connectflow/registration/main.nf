#!/usr/bin/env nextflow

if(params.help) {
    usage = file("$baseDir/USAGE")
    cpu_count = Runtime.runtime.availableProcessors()

    bindings = ["outlier_alpha":"$params.outlier_alpha",
                "cpu_count":"$cpu_count",
                "b_thr":"$params.b_thr",
                "b_thr":"$params.b_thr",
                "para_diff":"$params.para_diff",
                "perp_diff":"$params.perp_diff",
                "iso_diff":"$params.iso_diff"]

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
log.info "Label list: $params.label_list"
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
    antsRegistrationSyN.sh -d 3 -f ${native_anat} -m ${template} -t s -o ${sid}__output
    """
}

process Apply_transform {
    input:
    tuple val(sid), file(native_anat), file(atlas), file(affine), file(warp)

    output:
    tuple val(sid), file("${sid}__labels.nii.gz")

    script:
    """
    antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${native_anat} -o ${sid}__labels.nii.gz -n genericLabel -u int
    """
}

process Copy_sup_files {
    input:
    tuple val(sid), file(sup_files_1),file(sup_files_2),file(sup_files_3),file(sup_files_4),file(sup_files_5)

    output:
    tuple val(sid), file("$sup_files_1"),file("$sup_files_2"),file("$sup_files_3"),file("$sup_files_4"),file("$sup_files_5")

    script:
    """
    echo $sup_files_1 $sup_files_2
    """
}


workflow {
    /* Input files to fetch */
    root = file(params.input)
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    ref_images = Channel.fromPath("$root/*/*__ref_image.nii.gz").map{[it.parent.name, it]}
    info = Channel.fromFilePairs("$root/*/sub*__{dwi.bval,dwi.bvec,dwi.nii.gz,peaks.nii.gz,pft_tracking_prob_wm_seed_0.trk}", size: 5, flat: true)

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    ref_images.combine(template).set{data_registration}
    Register_Template_to_Ref(data_registration)

    /* Appy registration transformation to atlas  */
    ref_images.combine(atlas).join(Register_Template_to_Ref.out, by:0).set{data_transfo}
    Apply_transform(data_transfo)

    Copy_sup_files(info)
}