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

if (params.input){
    log.info "Input: $params.input"
    input = file(params.input)
    in_data = Channel
        .fromFilePairs("$input/**/*{brain_mask.nii.gz,bval,bvec,dwi.nii.gz}",
                       size: 4,
                       maxDepth:1,
                       flat: true) {it.parent.name}
}

log.info "TPIL msmt fODF pipeline"
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
log.info "[FRF]"
log.info "set_frf: $params.set_frf"
log.info ""
log.info "[FODF]"
log.info "sh_order: $params.sh_order"
log.info ""
log.info ""

log.info "Input: $params.input"

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
}

process Compute_FRF {

    input:
    tuple val(sid), file(brain_mask), file(bval), file(bvec), file(dwi)

    output:
    tuple val(sid), file("${sid}__frf_wm.txt"), file("${sid}__frf_gm.txt"), file("${sid}__frf_csf.txt")

    script:
    """
    echo $sid
    echo $dwi
    scil_compute_msmt_frf.py $dwi $bval $bvec ${sid}__frf_wm.txt ${sid}__frf_gm.txt ${sid}__frf_csf.txt
    """
}


process FODF_Metrics {

    input:
    tuple val(sid), file(brain_mask), file(bval), file(bvec), file(dwi), file(frf_wm), file(frf_gm), file(frf_csf)

    output:
    tuple val(sid), file("${sid}__fodf.nii.gz")

    script:
    """
    scil_compute_msmt_fodf.py $dwi $bval $bvec $frf_wm $frf_gm $frf_csf \
        --mask $brain_mask --sh_order $params.sh_order --sh_basis $params.basis
    """
}


workflow {
    Compute_FRF(in_data)
    in_data.join(Compute_FRF.out, by:0).set{data_for_fodf}
    FODF_Metrics(data_for_fodf)
}
