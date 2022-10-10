def run_roi_pipeline(sub_id):

    import os
    from functional_connectivity import save_roi_timeseries, compute_roi_fc, compute_graph_properties

    lsa_dir = '../../GLMs_fieldmaps_rp+fd+acc_csf/lsa'

    print(f'{sub_id}: fc & graph analyses running')

    # extract timeseries
    ts_fname = f'{lsa_dir}/roi_timeseries/{sub_id}_roi_timeseries.xlsx'
    if not os.path.exists(ts_fname):
        print(f'{sub_id}: extracting roi timeseries')
        save_roi_timeseries(sub_id)
    else:
        print(f'{sub_id}: already extracted roi timeseries')

    # compute functional connectivity 
    fc_fname = f'{lsa_dir}/roi_fc/{sub_id}_roi_fc.xlsx'
    if not os.path.exists(fc_fname):
        print(f'{sub_id}: correlating roi timeseries')
        compute_roi_fc(ts_fname, kind='correlation')
    else:
        print(f'{sub_id}: already correlated roi timeseries')

    # # compute graph properties
    # atlas = 'Shen'
    # graph_fname = f'{lsa_dir}/graphs/{sub_id}_unweighted_graph_{atlas}.pkl'
    # if not os.path.exists(graph_fname):
    #     print(f'{sub_id}: computing graph properties of timeseries correlations')
    #     compute_graph_properties(fc_fname, atlas=atlas)
    # else:
    #     print(f'{sub_id}: already computed graph properties of timeseries correlations')

    # # compute long-axis pattern similartity analysis
    # if not os.path.exists('../../Results/longaxis_ps_' + glm_name + '.xlsx'):
    #     run_ps_longaxis(glm_name)

    # # compute mds analysis
    # if not os.path.exists('../../Results/longaxis_stress_' + glm_name + '.xlsx'):
    #     run_mds_longaxis(glm_name)