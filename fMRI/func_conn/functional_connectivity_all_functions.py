import numpy as np
import pandas as pd

import networkx as nx 
import nibabel as nib

from nilearn import datasets
from nilearn.maskers import NiftiSpheresMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.decomposition import CanICA

from six.moves import cPickle as pickle 
def pickle_file(file_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f)
    f.close()
def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file

########################################################################################################
## timeseries
########################################################################################################

def get_brain_timeseries(img_fname, confounds=None, smoothing_fwhm=6, tr=None, 
                         detrend=True, standardize=True, low_pass=None, high_pass=None, verbose=False):
    '''
        https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiMasker.html#nilearn.maskers.NiftiMasker
        extract brain wide time series with shape = (n_volumes, n_voxels) 
        standardize=True -> zscore
    '''
    brain_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm, detrend=detrend, standardize=standardize,
                               low_pass=low_pass, high_pass=high_pass, t_r=tr, 
                               memory='nilearn_cache', memory_level=3, verbose=verbose)
    timeseries = brain_masker.fit_transform(img_fname, confounds=confounds)
    return timeseries, brain_masker

def get_sphere_avg_timeseries(img_fname, roi_coords, confounds=None, radius=8, tr=None, 
                              detrend=True, standardize=True, low_pass=None, high_pass=None, verbose=False):
    '''
        https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiMasker.html#nilearn.maskers.NiftiMasker
        extract mean time series from spherical roi with shape = (n_volumes, 1)
    '''
    sphere_masker = NiftiSpheresMasker(roi_coords, radius=radius, detrend=detrend, standardize=standardize,
                                     low_pass=low_pass, high_pass=high_pass, t_r=tr, 
                                     memory='nilearn_cache', memory_level=3, verbose=verbose)
    timeseries = sphere_masker.fit_transform(img_fname, confounds=confounds)
    return timeseries, sphere_masker

def get_timeseries(func_img, 
                   mask=None,
                   mask_type='gm-template',
                   radius=None,
                   target_shape=None,
                   target_affine=None,
                   smoothing_fwhm=None,
                   tr=None,
                   detrend=False, 
                   standardize=False, 
                   low_pass=None, high_pass=None,
                   confounds=None, 
                   standardize_confounds=False, 
                   high_variance_confounds=False,
                   memory_level=2,
                   verbose=0):
    
    '''
        mask_type:
        - roi, gm-template or whole-brain-template: shape = (n_voxels, n_volumes) 
            - if mask & func_img are diff resolution, func_img resampled to mask unless target_shape and/or target_affine are provided
            - for roi: pass in mask
        - sphere: extract mean time series from spherical roi, shape = (1, n_volumes)
            - coords should be passed into mask: [(x,y,z)] 
        - map: overlapping volumes, shape = (n_regions, n_volumes) 
        - atlas: non-overlapping volumes, shape = (n_regions, n_volumes)

        returns: timeseries of shape: (regions, timepoints)
    '''
    
    # to do: catch errors, deal w/ non-finites in sme sensible way... output info
                        
    if mask_type=='roi':
    
        masker = NiftiMasker(mask_img=mask,
                             smoothing_fwhm=smoothing_fwhm, t_r=tr,
                             detrend=detrend, standardize=standardize, 
                             low_pass=low_pass, high_pass=high_pass,
                             standardize_confounds=standardize_confounds,
                             high_variance_confounds=high_variance_confounds,
                             memory='nilearn_cache', memory_level=memory_level, verbose=verbose)   
        
    elif mask_type in ['whole-brain', 'gm']:

        masker = NiftiMasker(mask_strategy=mask_type + '-template',
                             smoothing_fwhm=smoothing_fwhm, t_r=tr,
                             detrend=detrend, standardize=standardize, 
                             low_pass=low_pass, high_pass=high_pass,
                             standardize_confounds=standardize_confounds,
                             high_variance_confounds=high_variance_confounds,
                             memory='nilearn_cache', memory_level=memory_level, verbose=verbose)    
        
    elif mask_type=='sphere':
        
        if radius is None: radius = 8 # default radius size
        masker = NiftiSpheresMasker(mask, radius=radius,
                                    smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                    detrend=detrend, standardize=standardize, 
                                    low_pass=low_pass, high_pass=high_pass,
                                    standardize_confounds=standardize_confounds,
                                    high_variance_confounds=high_variance_confounds,
                                    memory='nilearn_cache', memory_level=memory_level, verbose=verbose)   
    elif mask_type=='map':
        
        # https://nilearn.github.io/dev/auto_examples/03_connectivity/plot_signal_extraction.html
        masker = NiftiMapsMasker(maps_img=mask, resampling_target="data",
                                 mask_img=None,
                                 smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                 detrend=detrend, standardize=standardize, 
                                 low_pass=low_pass, high_pass=high_pass,
                                 standardize_confounds=standardize_confounds,
                                 high_variance_confounds=high_variance_confounds,
                                 memory='nilearn_cache', memory_level=memory_level, verbose=verbose)         
    elif mask_type=='atlas':
        
        # https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_probabilistic_atlas_extraction.html#sphx-glr-auto-examples-03-connectivity-plot-probabilistic-atlas-extraction-py
        masker = NiftiLabelsMasker(labels_img=mask, resampling_target="data",
                                   mask_img=None,
                                   smoothing_fwhm=smoothing_fwhm, t_r=tr,
                                   detrend=detrend, standardize=standardize, 
                                   low_pass=low_pass, high_pass=high_pass,
                                   standardize_confounds=standardize_confounds,
                                   high_variance_confounds=high_variance_confounds,
                                   memory='nilearn_cache', memory_level=memory_level, verbose=verbose)
        
    timeseries = masker.fit_transform(func_img, confounds=confounds)
    return timeseries.T, masker

########################################################################################################
## correlations between timeseries
########################################################################################################

def compute_roi_correlation(ts_fname, kind='correlation'):
    '''
        kind: correlation or partial-correlation
    '''
    kind = kind.replace('_', ' ') # partial_correlation -> partial correlation
    df   = pd.read_excel(ts_fname)
    rois = df['roi'].values
    ts   = df.iloc[:, 1:].values.T
    fc   = ConnectivityMeasure(kind=kind, vectorize=False, discard_diagonal=False).fit_transform([ts])
    fc_z = np.arctanh(fc.squeeze(0)) # outputs with an extra dimension for some reason
    fc_z = pd.DataFrame(fc_z, columns=rois, index=rois)

    fc_fname = ts_fname.replace('timeseries/', 'fc/') # folder name
    fc_fname = fc_fname.replace('timeseries', kind.replace('_', ' ') + '_z') # file name
    fc_z.to_excel(fc_fname) # output 

def compute_region_to_voxel_fc(sub_id, img_fname, out_dir, roi_type='roi', mask=None, mask_name=None, radius=8):

    # get region and whole brain correlations   
    brain_timeseries, brain_masker = get_timeseries(img_fname, mask_type='whole-brain') # timepoints, voxels
    if roi_type == 'sphere':
        region_timeseries, _ = get_timeseries(img_fname, mask=mask, mask_type=roi_type, radius=radius)
    elif roi_type == 'roi':
        region_timeseries, _ = get_timeseries(img_fname, mask=mask, mask_type=roi_type)
        region_timeseries    = np.mean(region_timeseries, 0).flatten() # average across voxels
        
    print(f'Region timeseries shape: {region_timeseries.shape}')
    print(f'Brain timeseries shape: {brain_timeseries.shape}')
    
    # perform the seed-to-voxel correlation
    corrs   = (np.dot(brain_timeseries, region_timeseries) / region_timeseries.shape[0]) # the inner dimensions must match for dot product
    corrs_z = np.arctanh(corrs) # fisher z-transform
    fc_img  = brain_masker.inverse_transform(corrs_z.T)
    print(f'FC image shape: {fc_img.shape}')
    
    # output image
    if roi_type == 'sphere':
        out_str = f'{mask[0][0]}_{mask[0][1]}_{mask[0][2]}_radius{radius}'
    elif roi_type == 'roi':
        out_str = mask_name
    fc_img.to_filename(f'{out_dir}/{sub_id}_{out_str}_correlation_z.nii.gz')

########################################################################################################
## network analysis
########################################################################################################

class graph_properties:

    def __init__(self, matrix, node_names, node_attributes=None, graph_attributes=None):

        '''
            upon initialization find graph properties
        '''
        # # Convert upper matrix to 2D matrix if upper matrix given
        # if matrix.ndim == 1:
        #     matrix = ut_vec_to_symm_mat(matrix)

        # generate graph and relabel nodes
        self.graph    = nx.from_numpy_matrix(matrix)
        index_mapping = dict(zip(np.arange(len(node_names)).astype(int), node_names))
        self.graph    = nx.relabel.relabel_nodes(self.graph, index_mapping)
        self.matrix   = matrix

        # Save graph/node/edge attributes
        if node_attributes is None: node_attributes = ['degree_centrality', 'betweenness_centrality', 
                                                       'closeness_centrality', 'eigenvector_centrality',
                                                       'clustering']
        for node_attr in node_attributes:
            print(f'Computing node attribute: {node_attr}')
            nx.set_node_attributes(self.graph, getattr(nx, node_attr)(self.graph), node_attr)            
                
        self.graph.attributes = {}
        if graph_attributes is None: graph_attributes = ['local_efficiency', 'global_efficiency']
                                                        # 'sigma', 'degree_assortativity_coefficient', 
                                                        #  'rich_club_coefficient']
        for graph_attr in graph_attributes:
            print(f'Computing graph attribute: {graph_attr}')
            if 'efficiency' in graph_attr:
                val = getattr(nx.algorithms.efficiency_measures, graph_attr)(self.graph)
            else:
                val = getattr(nx, graph_attr)(self.graph)
            self.graph.attributes[graph_attr] = val
        
    def find_communities(self, levels=1, assign=False):
        '''
            apply community detection algorithm
        '''
        comm_generator = nx.algorithms.community.girvan_newman(self.graph)
        for _ in np.arange(levels):
            comms = next(comm_generator)
        if assign:
            self.communities = comms
            counter = 1
            for comm in comms:
                for node_name in comm:
                    nx.set_node_attributes(self.graph, {node_name: counter}, "community")
                counter += 1
        return comms
    
def compute_graph_properties(fc_fname, atlas='HO', adj_ptile=0.95, weighted=False):
    
    # input
    df = pd.read_excel(fc_fname, index_col=0)
    rois = df.columns.values
    atlas_rois = [l for l in rois if  f'{atlas}_' in l]
    
    # get graph properties
    f = df.loc[atlas_rois, atlas_rois].values # extract only these fc values
    a = (f > np.percentile(f, adj_ptile)) # adjacency matrix thresholded with a percentile...
    if weighted: a = a * f 
    g = graph_properties(matrix=a, node_names=rois) 
    c = g.find_communities(levels=7)

    graph_data          = {}
    graph_data['fc']    = f 
    graph_data['adj']   = a 
    graph_data['graph'] = g
    graph_data['comms'] = c
    
    # output
    out_fname = fc_fname.replace('/roi_fc/', '/graphs/')
    if weighted: out_fname = out_fname.replace('roi_fc.xlsx', f'weighted_graph_{atlas}.pkl')
    else:        out_fname = out_fname.replace('roi_fc.xlsx', f'unweighted_graph_{atlas}.pkl')
    pickle_file(graph_data, out_fname)


########################################################################################################
## decomposition
########################################################################################################

def compute_CanICA_image(img_fnames, out_fname, mask=None, n_components=20, 
                         smoothing_fwhm=None, detrend=False, t_r=None, low_pass=None, high_pass=None, 
                         mask_strategy='whole-brain', do_cca=True):
    
    canica = CanICA(n_components=n_components,
                    smoothing_fwhm=smoothing_fwhm, # data might already be smoothed...
                    standardize=True,
                    detrend=detrend,
                    t_r=t_r,
                    low_pass=low_pass,
                    high_pass=high_pass,
                    mask=mask,
                    mask_strategy=mask_strategy + '-template', # 'whole-brain' or 'gm'
                    do_cca=do_cca, # not sure what this is about...
                    memory="nilearn_cache", memory_level=2, verbose=10, random_state=0)
    canica.fit(img_fnames)
    canica.components_img_.to_filename(out_fname)