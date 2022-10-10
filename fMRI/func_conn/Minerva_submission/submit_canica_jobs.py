import json, os, glob, shutil, sys 
import pandas as pd  
import subprocess, logging
import time

overwrite = False
base_dir  = '/sc/arion/projects/k23'
batch_dir = base_dir + '/code/func_conn/batch_dir'
glm_dir   = f'{base_dir}/GLMs_fieldmaps_rp+fd+acc_csf/lsa'
out_dir   = glm_dir + '/canica'
sub_info  = pd.read_excel(f'{base_dir}/participants_info.xlsx')

for group in ['HC','CD']:

    img_fnames = []
    incl_subs  = sub_info[(sub_info['dx']==group) & (sub_info['incl']==1)]['sub_id'].values

    for sub_dir in [sub for sub in glob.glob(glm_dir + '/subs/sub*')]:
        sub_id = int(sub_dir.split('sub-P')[-1])
        if sub_id in incl_subs: img_fnames.append(sub_dir + '/beta_4d.nii')

    out_fname = f'{out_dir}/canica_{group}_n{len(img_fnames)}.nii'
    # create the job  
    batch_script = f'{batch_dir}/batch_subs/{group}_canica.sh'
    with open(batch_script, 'w') as f:
        cookies = [ f'#!/bin/bash\n\n',
                    f'#BSUB -J canica_{group}\n',
                    f'#BSUB -P acc_guLab\n', 
                    f'#BSUB -q private\n', 
                    f'#BSUB -n 2\n',
                    f'#BSUB -W 01:00\n',
                    f'#BSUB -R rusage[mem=8000]\n',
                    f'#BSUB -o {batch_dir}/batch_output/nodejob-canica-{group}.out\n',
                    f'#BSUB -L /bin/bash\n\n',
                    f'ml python\n\n', 
                    f'cd {base_dir}/code/func_conn\n\n']
        f.writelines(cookies)  
        f.write(f"python -c 'import functional_connectivity as fc; fc.compute_CanICA_image(img_fnames=\"{img_fnames}\", out_fname=\"{out_fname}\")'")

    # submit the job
    if os.path.isfile(f'{out_dir}/{out_fname}'):
        logging.warning(f'{group} CanICA already completed!')
    else:
        logging.info(f'Submitting Job: {group}')
        subprocess.run(f'bsub < {batch_script}', shell=True)

    time.sleep(30)