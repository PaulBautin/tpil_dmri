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


process Run_COMMIT {
    input:
    tuple val(sid), file(bval), file(bvec), file(dwi), file(peaks), file(h5)

    output:
    tuple val(sid), file("${sid}__decompose_commit.h5")

    script:
    """
    echo $bval
    scil_run_commit.py $h5 $dwi $bval $bvec "${sid}__results_bzs/" --commit2 --processes 1
    mv "${sid}__results_bzs/commit_2/decompose_commit.h5" ./"${sid}__decompose_commit.h5"
    """
}


process Compute_Connectivity_without_similiarity {
    input:
    tuple val(sid), file(h5), file(labels), file(labels_list)

    output:
    tuple val(sid), file("*.npy")

    script:
    """
    scil_save_connections_from_hdf5.py $h5 ./bundles --node_keys 223
    scil_compute_connectivity.py $h5 $labels --force_labels_list $labels_list \
        --volume vol.npy --streamline_count sc.npy \
        --length len.npy --density_weighting
    scil_normalize_connectivity.py sc.npy sc_parcel_vol_normalized.npy \
        --parcel_volume $labels $labels_list
    scil_normalize_connectivity.py sc.npy sc_bundle_vol_normalized.npy \
        --bundle_volume vol.npy
    """
}

process Visualize_Connectivity {
    input:
    tuple val(sid), file(matrices), file(labels_list)

    output:
    tuple val(sid), file("*.png")

    script:
    String matrices_list = matrices.join(", ").replace(',', '')
    """
    for matrix in $matrices_list; do
        scil_visualize_connectivity.py \$matrix \${matrix/.npy/_matrix.png} --labels_list $labels_list --name_axis \
            --display_legend --histogram \${matrix/.npy/_hist.png} --nb_bins 50 --exclude_zeros --axis_text_size 5 5
    done
    """
}

process Connectivity_in_csv {
    input:
    tuple val(sid), file(matrices)

    output:
    tuple val(sid), file("*csv")

    script:
    String matrices_list = matrices.join("\",\"")
    """
    #!/usr/bin/env python3
    import numpy as np
    import os, sys

    for data in ["$matrices_list"]:
      fmt='%1.8f'
      if data == 'sc.npy':
        fmt='%i'

      curr_data = np.load(data)
      np.savetxt(data.replace(".npy", ".csv"), curr_data, delimiter=",", fmt=fmt)
    """
}

workflow {
    /* Input files to fetch */
    root = file(params.input)
    atlas = Channel.fromPath("$params.atlas")
    template = Channel.fromPath("$params.template")
    tractogram = Channel.fromPath("$root/*/*__tractogram.trk").map{[it.parent.name, it]}
    ref_images = Channel.fromPath("$root/*/*__ref_image.nii.gz").map{[it.parent.name, it]}
    dwi_data = Channel.fromFilePairs("$root/*/sub*__{dwi.bval,dwi.bvec,dwi.nii.gz,peaks.nii.gz}", size: 4, flat: true)
    label_list = Channel.fromPath("$params.label_list")

    main:
    /* Register template (same space as the atlas and same contrast as the reference image) to reference image  */
    ref_images.combine(template).set{data_registration}
    Register_Template_to_Ref(data_registration)

    /* Appy registration transformation to atlas  */
    ref_images.combine(atlas).join(Register_Template_to_Ref.out, by:0).set{data_transfo}
    Apply_transform(data_transfo)

    /* Create connectivity based on atlas and tractogram  */
    tractogram.combine(Apply_transform.out, by:0).set{data_connectivity}
    Decompose_Connectivity(data_connectivity)

    /* Create connectivity based on atlas and tractogram  */
    dwi_data.combine(Decompose_Connectivity.out, by:0).set{data_commit}
    data_commit.view()
    Run_COMMIT(data_commit)

    Run_COMMIT.out.combine(Apply_transform.out, by:0).combine(label_list).set{data_conn}
    data_conn.view()
    Compute_Connectivity_without_similiarity(data_conn)

    Compute_Connectivity_without_similiarity.out.combine(label_list).set{data_view}
    data_view.view()
    Visualize_Connectivity(data_view)

    Connectivity_in_csv(Compute_Connectivity_without_similiarity.out)

    /* Create ROI masks (based on atlas) for filtering tractogram
    Create_mask(Apply_transform.out)



    Filter tractogram based on ROI masks
    tractogram.join(Apply_transform.out, by:0).set{data_for_filtering}
    Filter_tractogram(data_for_filtering)

    Compute segmented bundle centroid
    Compute_Centroid(Filter_tractogram.out)

    Quality control of bundles
    ref_images.join(Filter_tractogram.out, by:0).set{bundles_for_qc}
    bundles_for_qc.view()
    bundle_QC(bundles_for_qc) */
}