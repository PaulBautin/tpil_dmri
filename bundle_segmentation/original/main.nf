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
log.info "Atlas: $params.atlas"
log.info "Template: $params.template"
log.info ""
log.info "[Filtering options]"
log.info "Outlier Removal Alpha: $params.outlier_alpha"
log.info "Source ROI: $params.source_roi"
log.info "Target ROI: $params.target_roi"
log.info ""

if (!(params.atlas) | !(params.template)) {
    error "You must specify an atlas and a template with command line flags. --atlas and --template, "
}

workflow.onComplete {
    log.info "Pipeline completed at: $workflow.complete"
    log.info "Execution status: ${ workflow.success ? 'OK' : 'failed' }"
    log.info "Execution duration: $workflow.duration"
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

process Apply_transform {
    input:
    tuple val(sid), file(affine), file(warp), file(native_anat), file(atlas)

    output:
    tuple val(sid), file("${sid}__atlas_transformed.nii.gz"), emit: atlas_transformed

    script:
    """
    antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${native_anat} -o ${sid}__atlas_transformed.nii.gz -n genericLabel -u int
    """
}

process Create_mask {
    input:
    tuple val(sid), path(atlas)

    output:
    tuple val(sid), file("${sid}__mask_source_*.nii.gz"), emit: masks_source
    tuple val(sid), file("${sid}__mask_target_*.nii.gz"), emit: masks_target

    script:
    """
    #!/usr/bin/env python3
    import nibabel as nib
    atlas = nib.load("$atlas")
    data_atlas = atlas.get_fdata()

    # Create masks target
    for s in $params.source_roi:
        mask = (data_atlas == s)
        mask_img = nib.Nifti1Image(mask.astype(int), atlas.affine)
        nib.save(mask_img, '${sid}__mask_target_'+str(s)+'.nii.gz')

    # Create masks target
    for t in $params.target_roi:
        mask = (data_atlas == t)
        mask_img = nib.Nifti1Image(mask.astype(int), atlas.affine)
        nib.save(mask_img, '${sid}__mask_source_'+str(t)+'.nii.gz')
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
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    tractogram_for_filtering = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.name, it]}
    ref_images = Channel.fromPath("$root/*/*__ref_image.nii.gz").map{[it.parent.name, it]}

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    ref_images.combine(template).set{data_registration}
    Register_Anat(data_registration)

    /* Appy registration transformation to atlas  */
    Register_Anat.out.transformations.join(ref_images, by:0).combine(atlas).set{data_transfo}
    Apply_transform(data_transfo)

    /* Create ROI masks (based on atlas) for filtering tractogram  */
    Create_mask(Apply_transform.out.atlas_transformed)

    /* Filter tractogram based on ROI masks  */
    masks_target = Create_mask.out.masks_target.transpose()
    masks_source = Create_mask.out.masks_source.transpose()
    tractogram_for_filtering.combine(masks_source.combine(masks_target, by:0), by:0).set{data_for_filtering}
    Clean_Bundles(data_for_filtering)

    Clean_Bundles.out.cleaned_bundle.combine(Register_Anat.out.transformations, by:0).combine(template).set{bundle_registration}
    Register_Bundle(bundle_registration)

    Register_Bundle.out.map{[it[1].name.split('_ses-')[1].split('_L')[0], it[1]]}.groupTuple(by:0).set{bundle_comparaison_inter}
    Bundle_Pairwise_Comparaison_Inter_Subject(bundle_comparaison_inter)

    Register_Bundle.out.map{[it[0].split('_ses')[0], it[1].name.split('__')[1].split('_L_')[0], it[1]]}.groupTuple(by:[0,1]).set{bundle_comparaison_intra}
    Bundle_Pairwise_Comparaison_Intra_Subject(bundle_comparaison_intra)

    Clean_Bundles.out.cleaned_bundle.combine(ref_images, by:0).set{bundles_for_screenshot}
    bundle_QC_screenshot(bundles_for_screenshot)
}
