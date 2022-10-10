#!/usr/bin/env python3

'''
    By Matthew Schafer, 2022
'''

import os, sys, glob, warnings, math, patsy
from pathlib import Path 
import pandas as pd 

user = os.path.expanduser('~')
base_dir = Path(f'{user}/Dropbox/Projects')

##########################################################################################
# decision info:
##########################################################################################

task_details = pd.read_excel(Path(f'{base_dir}/social_navigation_task/snt_details.xlsx'))
task_details.sort_values(by='slide_num', inplace=True)
decision_details = task_details[task_details['trial_type'] == 'Decision']

##########################################################################################
# defaults
##########################################################################################

character_roles  = ['first', 'second', 'assistant', 'powerful', 'boss'] # order of matlab outputs...
character_colors = ['green', 'blue', 'orange', 'purple', 'red']
group_colors     = ["#14C8FF", "#FD25A7"] # blue, pink - for plotting colors