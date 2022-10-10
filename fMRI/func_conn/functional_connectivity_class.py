class functional_connectivity:

    def __init__(self):
        
        print('Initializing functional connectivity class instance.')
        import numpy as np
        import pandas as pd
        from nilearn.maskers import NiftiSpheresMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker

    def get_timeseries(self, func_img, 
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
            
        self.mask_type  = mask_type
        self.timeseries = masker.fit_transform(func_img, confounds=confounds).T
        self.masker     = masker

        return self.timeseries
