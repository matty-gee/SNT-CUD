#!/usr/bin/env python3

import sys, os, glob, warnings
if not sys.warnoptions: warnings.simplefilter("ignore")

import nibabel as nib
import nilearn as nil
import pandas as pd
import numpy as np
from pathlib import Path

# # expects 'fMRI_tools' to be one directory up...
# cd   = str(Path(__file__).resolve())
# fsep = str(Path('/'))
# proj_dir = Path((fsep).join(cd.split(fsep)[0:-1])) # remove fname for project direcotry
# print(f'Project directory: {proj_dir}')
# tools_dir = Path(f'{(fsep).join(cd.split(fsep)[0:-2])}{fsep}fMRI_tools') 

# paths = ['..', proj_dir,
#          f'{tools_dir}/utilities_general', 
#          f'{tools_dir}/func_conn']

# [sys.path.insert(0, str(Path(p))) for p in paths if str(Path(p)) not in sys.path] # have to convert Path obj to string to add to system path
# # to remove: sys.path.remove(Path(p)) 

# # from snt_info import *
# from generic import read_excel, find_files, pickle_file, load_pickle, get_strings_matching_substrings
# from matrices import *
# from circ_stats import *
# from functional_connectivity import *

# parcellation_dir = Path(f'{tools_dir}/parcellations')

############################################################################################################
# data
############################################################################################################

# add data directories
data_dir = Path('/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_CUD')
if data_dir.exists():
    lsa_dir  = Path(f'{data_dir}/Analyses/GLMs_fieldmaps_rp/lsa_decision')
    beta_dir = Path(f'{lsa_dir}/images')
    fc_dir   = Path(f'{lsa_dir}/roi_fc')
    ts_dir   = Path(f'{lsa_dir}/roi_timeseries')
    beh_dir  = Path(f'{data_dir}/Data/Behavior')
    mask_dir = Path(f'{data_dir}/Masks')
else: print('Synapse not connected!')

try:    
    beh_df  = pd.read_excel(find_files(f'{data_dir}/Data/Summary', 'All-data_summary_n*.xlsx')[0])
    incl_df = pd.read_excel(find_files(data_dir, 'participants_qc_n*.xlsx')[0])
    beh_df  = incl_df[['sub_id','inclusion','memory_incl','fd_incl','other_incl']].merge(beh_df, on='sub_id')
    pmod_fnames = find_files(beh_dir, '*pmods*')
except: print(f'Behavioral data not found')


########################################################################################################
# snt plotting
########################################################################################################

# timing for decision trial epochs
decision_epochs = []
for on, off in zip(decision_details['onset'].values, decision_details['offset'].values):
    decision_epochs.extend(np.arange(int(np.round(on)), int(np.round(off))))


import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image
import matplotlib.patches as mpatches

# define plotting vraibles 
edgecolor = ".2"
tick_fontsize  = 10
label_fontsize = 13
title_fontsize = 15 #20
bar_width = 0.15 # this is a proportion of the total??
figsize = (5, 5)
facet_figsize = (5, 7.5)
roi_pal = sns.color_palette("Paired")
distance_pal = sns.color_palette("Purples")

character_colors = ['green', 'blue', 'orange', 'purple', 'red']
group_colors     = ["#14C8FF", "#FD25A7"] # blue, pink - for plotting colors

########################################################################################################
## project helper functions
########################################################################################################

def get_fname_ids(fnames, exclude=True, colname='fname'):
    ''' take a list of filenames, parse the subject id & return dataframe w/ subject info
    '''
    
    sub_ids = [float(Path(f).name.split('_')[0].replace('sub-P', '')) for f in fnames]
    df = pd.DataFrame([sub_ids, fnames]).T
    df.columns = ['sub_id', colname]

    # merge with other df
    df = incl_df[['inclusion','sub_id','dx','memory_incl','fd_incl','other_incl']].merge(df, on='sub_id')
    if exclude:
        df = df[df['inclusion']==1]
        df.reset_index(inplace=True, drop=True)
        print(f'included n={len(df)}')
    else:
        print(f'n={len(df)}')
    return df

def check_labels(labels_check, labels):
    '''
        flip the order of sublabels, if needed, in strings to match columnns 
    '''
    labels_out = []
    for l in labels_check:
        if l in labels:
            labels_out.append(l)
        else:
            l_ = l.split('_and_')
            labels_out.append(l_[1] + '_and_' + l_[0])
    return labels_out

def get_fc_labels(region_list, df):
    
    atlas_fcs     = [c for c in df.columns if '_and_' in c]
    atlas_regions = np.unique([c.split('_and_') for c in atlas_fcs])
    regions_      = get_strings_matching_substrings(atlas_regions, region_list)
    fcs_          = symm_mat_labels_to_vec(regions_, upper=True)
    fcs_          = check_labels(fcs_, atlas_fcs)
    print(f'number of regions found = {len(regions_)}')
    print(f'number of func conn found = {len(fcs_)}')
    assert len(regions_) * (len(regions_) - 1) / 2 == len(fcs_), \
            'Maybe missing fcs? Number of fcs != bottom triangle count'
            
    return [regions_, fcs_]