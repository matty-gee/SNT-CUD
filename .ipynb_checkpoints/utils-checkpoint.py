import os.path
from collections import Counter
from re import search
from pathlib import Path
from datetime import date
today = date.today()

# sci computing stuff
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

########################################################################################################################
# exclusions
########################################################################################################################

repl_excl = ['18006', '20002','20003','20005','21002','21008','21012','21020','22001'] 
orig_excl = ['13', '11', '4']
excl = orig_excl + repl_excl

########################################################################################################################
# define some task stuff
########################################################################################################################

task_details = pd.read_excel('task_details.xls') 
task_details.sort_values(by=['slide_num']) # make sure to sort
decision_details = task_details[task_details['trial_type']=='Decision']

character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss']
character_colors = ['r','g','b','y','m']
character_labels = task_details[task_details['trial_type']=='Decision'].sort_values(by=['slide_num'])['role_num'].values
character_labels = character_labels[character_labels != 9.0] # exclude neutrals

########################################################################################################################
# generic helper functions
########################################################################################################################

# PICKLED FILES
from six.moves import cPickle as pickle 
def pickle_file(file_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f)
    f.close()
def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        ret_file = pickle.load(f)
    return ret_file

# JSON FILES
import json
from json import JSONEncoder
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=json_encoder)

class json_encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return JSONEncoder.default(self, obj)

########################################################################################################################
# neutral character
########################################################################################################################

def remove_neutrals(arr):
    '''
        remove the neutral trials from array: trials 15,16,36
    '''
    return np.delete(arr, np.array([14,15,35]), axis=0) 

def add_neutrals(arr, add=[0,0]):
    '''
        add values for neutral trials into array; trials 15,16,36
        inputted array should have length 60; outputted will have length 63
    '''
    temp = arr.copy()
    for row in [14,15,36]:
        temp = np.insert(temp, row, add, axis= 0)
    return temp

########################################################################################################################
# dataframes
########################################################################################################################

def sort_reset(df, by='sub_id'):
    df.sort_values(by, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def finite_mask(df):
    return df[np.isfinite(df)]

def df_subset_on_cols(df, subset_dict):
    for col,value in subset_dict.items():
        df = df[df[col] == value]
    if 'sub_id' in df.columns:
        df = df.sort_values('sub_id').reset_index(drop=True)
    return df

def digitize_df(df, cols=[], n_bins=10, zscored=True):
    df_ = df.copy()
    if len(cols) == 0:
        cols = df_.columns
    for col in cols:
        bins_ = stats.mstats.mquantiles(df[col], np.linspace(0., 1.0, num=n_bins, endpoint=False)) 
        df_[col] = digitized = np.digitize(df[col], bins_, right=True)
    return df_
    
########################################################################################################################
# manipulating matrices
########################################################################################################################

def mask_trials_out(trial_ixs):
    '''
        flexible masking for trialwise matrices...
    '''
    mask_rdm = np.ones((63,63))
    for ix in trial_ixs:
        mask_rdm[ix,:] = 0
        mask_rdm[:,ix] = 0
    return symm_mat_to_ut_vec(mask_rdm)

def digitize_rdm(rdm_raw, n_bins=10): 
    """
        Digitize an input matrix to n bins (10 bins by default)
        rdm_raw: a square matrix 
    """
    rdm_bins = [np.percentile(np.ravel(rdm_raw), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
    rdm_vec_digitized = np.digitize(np.ravel(rdm_raw), bins = rdm_bins) * (100 // n_bins) # Compute the vectorized digitized value 
    rdm_digitized = np.reshape(rdm_vec_digitized, np.shape(rdm_raw)) # Reshape to matrix
    rdm_digitized = (rdm_digitized + rdm_digitized.T) / 2     # Force symmetry in the plot
    return rdm_digitized
 
def symm_mat_to_ut_vec(mat):
    """
        go from symmetrical matrix to vectorized/flattened upper triangle
    """
    vec_ut = mat[np.triu_indices(len(mat), k=1)]
    return vec_ut
def ut_mat_to_symm_mat(mat):
    '''
        go from upper tri matrix to symmetrical matrix
    '''
    for i in range(0, np.shape(mat)[0]):
        for j in range(i, np.shape(mat)[1]):
            mat[j][i] = mat[i][j]
    return mat
def ut_vec_to_symm_mat(vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat = ut_vec_to_ut_mat(vec)
    symm_mat = ut_mat_to_symm_mat(ut_mat)
    return symm_mat
def ut_vec_to_ut_mat(vec):
    '''
        go from vectorized/flattened upper tri to a upper tri matrix
            1. solve get matrix size: matrix_len**2 - matrix_len - 2*vector_len = 0
            2. then populate upper tri of a m x m matrix with the vector elements 
    '''
    
    # solve quadratic equation to find size of matrix
    from math import sqrt
    a = 1; b = -1; c = -(2*len(vec))   
    d = (b**2) - (4*a*c) # discriminant
    roots = (-b-sqrt(d))/(2*a), (-b+sqrt(d))/(2*a) # find roots   
    if False in np.isreal(roots): # make sure roots are not complex
        raise Exception('Roots are complex') # dont know if this can even happen if not using cmath...
    else: 
        m = int([root for root in roots if root > 0][0]) # get positive root as matrix size
        
    # fill in the matrix 
    mat = np.zeros((m,m))
    vec = vec.tolist() # so can use vec.pop()
    c = 0  # excluding the diagonal...
    while c < m-1:
        r = c + 1
        while r < m: 
            mat[c,r] = vec[0]
            vec.pop(0)
            r += 1
        c += 1
    return mat

def get_pw_dist_ut_vec(coords, metric='euclidean'):
    return symm_mat_to_ut_vec(pairwise_distances(coords, metric=metric))
def remove_diag(arr):
    arr = arr.copy()
    np.fill_diagonal(arr, np.nan)
    return arr[~np.isnan(arr)].reshape(arr.shape[0], arr.shape[1] - 1)

########################################################################################################################
# combos, permutations & derangements
########################################################################################################################

def combos(arr, r):
    """
        get combos
        args: 
            arr: np.array to get combos from
            r: len of combos 
    """
    from itertools import combinations 
    return list(combinations(arr, r))

def get_unique_permutations(seq):
    """
    Yield only unique permutations of seq in an efficient way.

    A python implementation of Knuth's "Algorithm L", also known from the 
    std::next_permutation function of C++, and as the permutation algorithm 
    of Narayana Pandita.
    """

    # Precalculate the indices we'll be iterating over for speed
    i_indices = list(range(len(seq) - 1, -1, -1))
    k_indices = i_indices[1:]

    # The algorithm specifies to start with a sorted version
    seq = sorted(seq)

    while True:
        yield seq

        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # Introducing the slightly unknown python for-else syntax:
            # else is executed only if the break statement was never reached.
            # If this is the case, seq is weakly decreasing, and we're done.
            return

        # Get item from sequence only once, for speed
        k_val = seq[k]

        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break

        # Swap them in the most efficient way
        (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                           # 0 0 1 1 1 1 0 0

        # Reverse the part after but not                           k
        # including k, also efficiently.                     0 0 1 1 0 0 1 1
        seq[k + 1:] = seq[-1:k:-1]

import copy
def get_derangements(x):
    return np.array([copy.copy(s) for s in get_unique_permutations(x) if not any([a == b for a, b in zip(s, x)])])

########################################################################################################################
## behavioral rdms
########################################################################################################################

def add_neutrals_coords(coords):
    '''
        add neutral trials (15,16,36) into behavior
        60->63
    '''
    for trial in np.array([14,15,35]):
        coords = np.insert(coords, trial, np.array((0, 0)), 0)
    return coords

def remove_neutrals(arr):
    """
        remove the neutral trials from an array
    """
    return np.delete(arr, np.array([14,15,35]), axis=0) # trials 15,16,36

########################################################################################################################
## plotting stuff
########################################################################################################################

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


#############################################
# PATTERN SIMILARITY PLOT
#############################################


 


    