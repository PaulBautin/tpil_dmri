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
    tuple val(sid), path(t1_image)

    output:
    tuple val(sid), file("${sid}__t1-L_Accu_first.nii.gz")

    script:
    """
    run_first_all -d -i ${t1_image} -o ${sid} -b
    """
}

process Clean_Bundles {
    memory_limit='6 GB'

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
    tractogram_for_filtering = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.parent.name, it]}
    t1_images = Channel.fromPath("$root/*/Register_T1/*__t1_warped.nii.gz").map{[it.parent.parent.name, it]}
    t1_images.view()

    main:
    /* Create ROI masks (based on atlas) for filtering tractogram  */
    Create_sub_mask(t1_images)
}
