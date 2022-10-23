import json, os, glob, shutil, sys   
import subprocess, logging
import time

overwrite  = False
base_dir   = '/sc/arion/projects/k23'
batch_dir  = base_dir + '/code/func_conn/batch_dir'
glm_dir    = f'{base_dir}/GLMs_fieldmaps_rp+fd+acc_csf/lsa'
out_dir    = glm_dir + '/roi_timeseries'

for sub_dir in [sub for sub in glob.glob(glm_dir + '/subs/sub*')]:

    sub_id = sub_dir.split('sub-P')[1]

    # create the job  
    batch_script = f'{batch_dir}/batch_subs/{sub_id}_timeseries.sh'
    with open(batch_script, 'w') as f:
        cookies = [ f'#!/bin/bash\n\n',
                    f'#BSUB -J timeseries_{sub_id}\n',
                    f'#BSUB -P acc_guLab\n', 
                    f'#BSUB -q private\n', 
                    f'#BSUB -n 1\n',
                    f'#BSUB -W 00:15\n',
                    f'#BSUB -R rusage[mem=8000]\n',
                    f'#BSUB -o {batch_dir}/batch_output/nodejob-timeseries-{sub_id}.out\n',
                    f'#BSUB -L /bin/bash\n\n',
                    f'ml python\n\n', 
                    f'cd {base_dir}/code/func_conn\n\n']
        f.writelines(cookies)  
        f.write(f"python -c 'import functional_connectivity as fc; fc.get_timeseries_xlsx(sub_dir=\"{sub_dir}\", out_dir=\"{out_dir}\")'")

    # submit the job
    if os.path.isfile(f'{out_dir}/{sub_id}_roi_timeseries.xlsx'):
        f'{out_dir}/{sub_id}_roi_timeseries.xlsx'
        logging.warning(f'{sub_id} already completed!')

        if not overwrite:
            logging.info(f'Skipping {sub_id}')
            continue
        else:
            logging.warning(f'Re-running {sub_id}, and overwriting results!')

    logging.info(f'Submitting Job: {sub_id}')
    subprocess.run(f'bsub < {batch_script}', shell=True)
    time.sleep(30)
