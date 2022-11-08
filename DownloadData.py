import subprocess
options = ['filt_global', 'filt_noglobal', 'nofilt_global']
strategy = ['reho', 'rois_aal', 'lfcd', 'rois_tt']
for opt in options:
    for strat in strategy:
        comm = "python download_abide_preproc.py -d "+strat+" -p cpac -s "+opt+" -o E:\\AbideI\\cpac\\rois_tt"
        print(comm)
        #break
        subprocess.run(comm)
