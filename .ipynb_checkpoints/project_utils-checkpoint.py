import sys, os, glob, warnings
warnings.filterwarnings("ignore") 

## load packages
user = os.path.expanduser('~')
proj_dir = user + '/Dropbox/Projects/CUD'
code_dir = user + '/Dropbox/Projects/Code'
for path in ['..', f'{proj_dir}/Code/fMRI/GLMs/2nd_level', 
            f'{proj_dir}/Code/fMRI/func_conn', f'{code_dir}/utilities', 
            f'{code_dir}/snt_behavior/preprocessing']:
    sys.path.insert(0, path)

from standard_modules import *
from generic import pickle_file, load_pickle, read_excel
from matrices import *
from circ_stats import *
from regression import * 
from classification import * 
import plotting as plot
import functional_connectivity as fc
import second_level

## define directories
syn_dir  = '/Volumes/synapse/projects/SocialSpace/Projects/SNT-fmri_CUD'
if os.path.exists(syn_dir):
    lsa_dir  = syn_dir + '/Analyses/GLMs_fieldmaps_rp+fd+csf/lsa_decision'
    beta_dir = f'{lsa_dir}/beta_images'
    fc_dir   = lsa_dir + '/roi_fc'
    ts_dir   = lsa_dir + '/roi_timeseries'
    beh_dir  = syn_dir + '/Data/Behavior/Pmod_analyses'
    mask_dir = syn_dir + '/Masks'
    pmod_fnames = glob.glob(beh_dir + '/*pmods*')
else:
    print('Synapse not connected - assuming timeseries and roi fcs are on Deskop/CUD')
    fc_dir = user + '/Desktop/lsa_decision/roi_fc'
    ts_dir = user + '/Desktop/lsa_decision/roi_timeseries'

## load behavioral data
try:    
    beh_df  = pd.read_excel(glob.glob(f'{syn_dir}/Data/Summary/All-data_summary_n*.xlsx')[0])
    incl_df = pd.read_excel(glob.glob(f'{syn_dir}/participants_qc_n*.xlsx')[0])
    beh_df  = incl_df[['sub_id','inclusion','memory_incl','fd_incl','other_incl']].merge(beh_df, on='sub_id')
except: 
    print(f'Behavioral data not found')

## task info
from snt_info import *

# timing for decision trial epochs
decision_epochs = []
for on, off in zip(decision_details['onset'].values, decision_details['offset'].values):
    decision_epochs.extend(np.arange(int(np.round(on)), int(np.round(off))))

##################################################
## project helper functions
##################################################

def get_fname_ids(fnames, exclude=True):
    '''
        return dataframe w/ sub info
    '''
    sub_ids = [f.split('/')[-1].split('_')[0] for f in fnames]
    sub_ids = [float(s.replace('sub-P', '')) for s in sub_ids]
    df = pd.DataFrame([sub_ids, fnames]).T
    df.columns = ['sub_id', 'fname']

    # merge with other df
    incl_df = pd.read_excel(glob.glob(f'{lsa_dir}/participants_qc_n*.xlsx')[0])
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

##################################################
## generic helper functions - refine and add to code base
##################################################

def get_strings_matching_substrings(strings, substrings):
    '''
        find strings in list that partially match any string in a list
    '''
    mask = [any(ss in s for ss in substrings) for s in strings]
    return list(np.array(strings)[mask])

def sort_symm_mat(mat, vec):
    '''
        Sorts rows/columns of a symmetrical matrix according to a separate vector.
    '''
    inds = vec.argsort()
    mat_sorted = mat.copy()
    mat_sorted = mat_sorted[inds, :]
    mat_sorted = mat_sorted[:, inds]
    return mat_sorted

def bootstrap_subject_matrix(similarity_matrix, random_state=None):
    '''
        shuffles subjects within a similarity matrix based on recommendation by Chen et al., 2016
    '''
    rs = sklearn.utils.check_random_state(random_state)
    n_sub = similarity_matrix.shape[0]
    bootstrap = sorted(rs.choice(np.arange(n_sub), size=n_sub, replace=True))
    return similarity_matrix[bootstrap, :][:, bootstrap]