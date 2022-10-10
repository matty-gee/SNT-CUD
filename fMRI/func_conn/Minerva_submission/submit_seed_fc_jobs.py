import json, os, glob, shutil, sys   
import subprocess, logging
import time

overwrite  = False

base_dir   = '/sc/arion/projects/k23'
batch_dir  = f'{base_dir}/code/func_conn/batch_dir'
mask_dir   = f'{base_dir}/Masks'
glm_dir    = f'{base_dir}/GLMs_fieldmaps_rp+fd+csf/lsa_decision'
out_dir    = f'{glm_dir}/connectivity_images'
if not os.path.exists(out_dir): os.mkdir(out_dir)

# for seed based
coords = [[(-22, -15, -18)], [(51, 50, 1)]]
radius = 8

# for mask based
masks = {'LHPC_ant': f'{mask_dir}/L-HPC_ant_harvardoxford_maxprob-thr25-1mm.nii',
         'LHPC_mid': f'{mask_dir}/L-HPC_mid_harvardoxford_maxprob-thr25-1mm.nii',
         'LHPC_post': f'{mask_dir}/L-HPC_post_harvardoxford_maxprob-thr25-1mm.nii'}

for roi, mask_fname in masks.items():

    sub_dirs = [sub for sub in glob.glob(glm_dir + '/subs/sub*')]
    sub_dirs.sort()
    for sub_dir in sub_dirs:

        sub_id     = sub_dir.split('/')[-1]
        # coords_str = str(coords[0][0]) + '_' + str(coords[0][1]) + '_' + str(coords[0][2]) 
        img_fname  = sub_dir + '/beta_4d.nii'

        # create the job  
        batch_script = f'{batch_dir}/batch_subs/{sub_id}_region-to-voxel.sh'
        with open(batch_script, 'w') as f:
            cookies = [ f'#!/bin/bash\n\n',
                        f'#BSUB -J region-to-voxel_{sub_id}\n',
                        f'#BSUB -P acc_k23\n', 
                        f'#BSUB -q express\n', 
                        f'#BSUB -n 2\n',
                        f'#BSUB -W 00:15\n',
                        f'#BSUB -R rusage[mem=8000]\n',
                        f'#BSUB -o {batch_dir}/batch_output/nodejob-region-to-voxel-{sub_id}.out\n',
                        f'#BSUB -L /bin/bash\n\n',
                        f'ml python\n\n', 
                        f'cd {base_dir}/code/func_conn\n\n']
            f.writelines(cookies)
            f.write(f"python -c 'import functional_connectivity as fc;\
                                 fc.compute_region_to_voxel_fc(sub_id=\"{sub_id}\",\
                                 img_fname=\"{img_fname}\",\
                                 mask=\"{mask_fname}\", mask_name=\"{roi}\",\
                                 out_dir=\"{out_dir}\")'")

        # submit the job
        if os.path.isfile(f'{out_dir}/{sub_id}_{roi}_correlation_z.nii.gz'):
            logging.warning(f'{sub_id} already completed!')
            if not overwrite:
                logging.info(f'Skipping {sub_id}')
                continue
            else:
                logging.warning(f'Re-runningg {sub_id}, and overwriting results!')

        logging.info(f'Submitting Job: {sub_id}')
        subprocess.run(f'bsub < {batch_script}', shell=True)
        time.sleep(30)
