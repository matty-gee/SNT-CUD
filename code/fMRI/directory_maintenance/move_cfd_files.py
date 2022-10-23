import glob, os, shutil

base_dir = '/sc/arion/projects/k23/derivatives_fieldmaps'
out_dir  = base_dir + '/cfd_files' 
if not os.path.exists(out_dir): os.mkdir(out_dir)
for sub_dir in [d for d in glob.glob(base_dir + '/fmriprep/sub*') if '.html' not in d]:
    sub_id = sub_dir.split('P')[-1]
    print(sub_id)
    for fname in ['*confounds_timeseries.json', '*confounds_timeseries.tsv']:
        try: 
            inpath  =  glob.glob(f'{sub_dir}/func/{fname}')[0]
            fname   = inpath.split('/')[-1]
            outpath = f'{out_dir}/{fname}'
            shutil.copy(inpath, outpath) 
        except:
            print(f'ERROR: {sub_id}')
