import os, glob, time, subprocess, logging

nodes     = 'public'
base_dir  = '/sc/arion/projects/k23'
batch_dir = f'{base_dir}/code/func_conn/batch_dir'

if nodes == 'Gu':
    project = 'acc_guLab'
    queue   = 'private'
else:
    project = 'acc_k23'
    queue   = 'premium'    

for nuisance in ['rp+fd', 'rp']:
    for bold_event in ['decision']:

        lsa_dir  = f'{base_dir}/GLMs_fieldmaps_{nuisance}/lsa_{bold_event}'
        sub_dirs = [sub for sub in glob.glob(lsa_dir + '/subs/sub*')]
        sub_dirs.sort()

        for sub_dir in sub_dirs:

            # create the job
            sub_id = sub_dir.split('sub-P')[-1]
            batch_script = f'{batch_dir}/batch_subs/{sub_id}_roi_pipeline.sh'
            with open(batch_script, 'w') as f:
                cookies = [ f'#!/bin/bash\n\n',
                            f'#BSUB -J {sub_id}_roi\n',
                            f'#BSUB -P {project}\n', 
                            f'#BSUB -q {queue}\n', 
                            f'#BSUB -n 1\n',
                            f'#BSUB -W 1:00\n',
                            f'#BSUB -R rusage[mem=10000]\n',
                            f'#BSUB -o {batch_dir}/batch_output/nodejob-roi_pipeline-{sub_id}.out\n',
                            f'#BSUB -L /bin/bash\n\n',
                            f'ml python\n\n',
                            f'cd {base_dir}/code/func_conn\n\n']
                f.writelines(cookies)  
                f.write(f"python -c 'import pipelines as pl; \
                        pl.run_roi_pipeline(sub_id=\"{sub_id}\",\
                                            lsa_dir=\"{lsa_dir}\")'")

            # submit the job
            out_fname = f'{lsa_dir}/roi_timeseries/{sub_id}_roi_timeseries.xlsx'
            # if not os.path.exists(out_fname):
            logging.info(f'Submitting job for {sub_id}')
            subprocess.run(f'bsub < {batch_script}', shell=True)
            time.sleep(20)