import os, glob

preprc_dir = '/sc/arion/projects/k23/derivatives_fieldmaps/fmriprep/'
for sub_dir in [d for d in glob.glob(preprc_dir + '/sub*') if '.html' not in d]:
    print(sub_dir)
    sub_id = sub_dir.split('/')[-1]
    cfd_fnames = []
    for f in ['*_rp*.txt', '*wmcsf-thr95.csv', '*wmcsf-thr95.txt', 'rp.txt', 'rp+fd+acc_csf.txt']:
        try:
            cfd_fnames.append(glob.glob(f'{sub_dir}/func/{f}')[0])
        except:
            continue
    print(sub_id, cfd_fnames)
    for fname in cfd_fnames:
        os.remove(fname)