#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
'''

def run_roi_pipeline(sub_id, lsa_dir):

    '''
    '''

    import os, shutil
    import functional_connectivity as fc
    
    print(f'{sub_id}: fc & graph analyses running')

    ## make paths if needed
    for path in ['images','roi_timeseries', 'roi_fc', 'graphs', 'connectivity_images']:
        if not os.path.exists(f'{lsa_dir}/{path}'):
            os.mkdir(f'{lsa_dir}/{path}')

    ## move images
    for img_fname in ['beta_4d.nii', 'mask.nii']:
        in_fname  = f'{lsa_dir}/subs/{sub_id}/{img_fname}'
        out_fname = f'{lsa_dir}/images/{sub_id}/{sub_id}_{img_fname}'
        if not os.path.exists(out_fname):
            try:    shutil.copy(in_fname, out_fname) 
            except: print(f'error with {sub_id}: {img_fname}')
    
    ## extract timeseries
    ts_fname = f'{lsa_dir}/roi_timeseries/{sub_id}_roi_timeseries.xlsx'
    if not os.path.exists(ts_fname):
        print(f'{sub_id}: extracting roi timeseries')
        fc.save_roi_timeseries(sub_id, lsa_dir=lsa_dir)
    else:
        print(f'{sub_id}: already extracted roi timeseries')

    ## compute functional connectivity 
    # -- can try tangent too... but this requires all of the group images 
    for corr_kind in ['correlation', 'partial_correlation']:
        fc_fname = f'{lsa_dir}/roi_fc/{sub_id}_roi_{corr_kind}_z.xlsx'
        if not os.path.exists(fc_fname):
            print(f'{sub_id}: calculating functional connectivity with {corr_kind}')
            fc.compute_roi_correlation(ts_fname, kind=corr_kind)
        else:
            print(f'{sub_id}: already calculated functional connectivity with {corr_kind}')

        ## compute graph properties of func connectivity
        atlas = 'HO'
        graph_fname = f'{lsa_dir}/graphs/{sub_id}_{atlas}_{corr_kind}_z_unweighted_graph.pkl'
        if not os.path.exists(graph_fname):
            print(f'{sub_id}: computing graph properties of timeseries correlations')
            fc.compute_graph_properties(fc_fname, atlas=atlas)
        else:
            print(f'{sub_id}: already computed graph properties of timeseries correlations')

    # # compute long-axis pattern similartity analysis
    # if not os.path.exists('../../Results/longaxis_ps_' + glm_name + '.xlsx'):
    #     run_ps_longaxis(glm_name)

    # # compute mds analysis
    # if not os.path.exists('../../Results/longaxis_stress_' + glm_name + '.xlsx'):
    #     run_mds_longaxis(glm_name)

    ## seed or region-based wholebrain connectivity
    mask_dir = '/sc/arion/projects/k23/Masks'
    out_dir  = f'{lsa_dir}/connectivity_images'
    masks = {'LHPC_ant': f'{mask_dir}/L-HPC_ant_harvardoxford_maxprob-thr25-1mm.nii',
            'LHPC_mid': f'{mask_dir}/L-HPC_mid_harvardoxford_maxprob-thr25-1mm.nii',
            'LHPC_post': f'{mask_dir}/L-HPC_post_harvardoxford_maxprob-thr25-1mm.nii'}

    for roi, mask_fname in masks.items():
        if not os.path.isfile(f'{out_dir}/{sub_id}_{roi}_correlation_z.nii.gz'):
            fc.compute_region_to_voxel_fc(sub_id=sub_id,
                                          img_fname=img_fname,
                                          mask=mask_fname, mask_name=roi,
                                          out_dir=out_dir)
