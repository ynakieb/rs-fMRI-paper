'''
The goal of this code is to create dynamic connectivity feature Files dFC.
1. Calculate dynamic Correlation Matricies, using gaussian kernel
2. Create Feature tables (.csv) for each subject, labeled by actual area names
'''
import pandas as pd
import numpy as np
import concurrent.futures
import os
from scipy.signal.windows import gaussian
import warnings
from joblib import Parallel, delayed
import argparse

sub_pth_tt = 'Caltech_0051456_rois_tt.csv'
sub_pth_aal = 'Caltech_0051456_rois_aal.csv'
main_path = 'E:\\AbideI\\ProcessedData\\'

filt_pth = ['nofilt_noglobal_', 'filt_global_', 'filt_noglobal_', 'nofilt_global_']
atl_pth = ['rois_aal\\', 'rois_tt\\']
# in_folders = [main_path+o+s+'feat\\' for o in filt_pth for s in atl_pth]
# out_folder = [main_path+o+s+'corr\\' for o in filt_pth for s in atl_pth]


def create_corr_names(col_names):
    col_nam = []
    n_areas = len(col_names)
    for i in range(n_areas):
        for j in range(i+1, n_areas):
            col_nam.append( '__'.join([col_names[i],col_names[j]] )+'_wk' )
            col_nam.append( '__'.join([col_names[i],col_names[j]] )+'_st' )
    return col_nam


def parallelize_corr(values, window, step, ind_i):
    n_feat, n_areas = values.shape
    w = len(window)

    # for ind_i in range(n_areas):
    # For each brain region:
    # series = df.iloc[:,ind_i].values
    res = np.zeros(2*len(range(ind_i + 1, n_areas)))
    cntr_res = 0
    for ind_j in range(ind_i + 1, n_areas):
        n_wk_cntr = 0
        n_st_cntr = 0
        cntr = 0
        for i in range(0, n_feat - w, step):
            val = np.corrcoef(window * values[i:i + w, ind_i], window * values[i:i + w, ind_j])[
                0, 1]  # the covar matrix of 2x2
            cntr += 1
            if abs(val) <= 0.25: n_wk_cntr += 1
            if abs(val) >= 0.8: n_st_cntr += 1
        n_wk = n_wk_cntr / cntr
        n_st = n_st_cntr / cntr
        # print(arr_key, n_wk, n_st)
        # corr_feat.loc[0, arr_key + '_wk'] = n_wk
        # corr_feat.loc[0, arr_key + '_st'] = n_st
        res[cntr_res] = n_wk
        res[cntr_res+1] = n_st
        cntr_res += 2
    return res


def create_dFC(f_path,window=None, step=1):
    # previous : df, window=None, step=1
    # corr_col_names: the OUTPUT col names , created by create_corr_names
    '''
    Creates The values for just one subject df'''
    # w needs to be an odd number
    if window is None:
        window = gaussian(21, std=3)
    w = len(window)
    output_path = os.path.join(f_path ,'corr')
    FolderPath = os.path.join(f_path ,'feat')
    print(output_path)
    print(FolderPath)
    if FolderPath.split('/')[-2][-3:] =='aal':
        sub_pth = 'Caltech_0051456_rois_aal.csv'
    elif FolderPath.split('/')[-2][-2:] =='tt':
        sub_pth = 'Caltech_0051456_rois_tt.csv'
    else:
        warnings.warn('Unable to Get the sub_path')

    ex_df = pd.read_csv(os.path.join(FolderPath, sub_pth))
    corr_col_names = create_corr_names(ex_df.columns)
    subjects_corr_df = pd.DataFrame(columns=corr_col_names)
    subjs_fldrs = [os.path.join(FolderPath, f) for f in os.listdir(FolderPath)]

    for filepath in subjs_fldrs:
        subj = filepath.split('/')[-1].split('.')[0]  # ex: Caltech_0051456_rois_tt

        #         if os.path.exists(output_path + subj+ '_corr.csv'): #unhash if needed
        #             print(f'Subj %s already exists, continuing')
        #             continue
        # sub_name = ('_').join(subj.split('_')[:2])  # [not used] without the atlas name, to be used as index, ex: Caltech_0051456_rois

        df = pd.read_csv(filepath)
        n_feat, n_areas = df.shape
        values = df.values
        with Parallel(n_jobs=20) as para:
            res = para(delayed(parallelize_corr)(values, window, step, ind_i) for ind_i in range(n_areas))
            subjects_corr_df.loc[subj, :] = np.concatenate(res)
        # return corr_feat
    #     for ind_i in range(n_areas):
    #         # For each brain region:
    #         # series = df.iloc[:,ind_i].values
    #         # print(ind_i, end=' ')
    #         for ind_j in range(ind_i + 1, n_areas):
    #             temp_corr_arr = []
    #             for i in range(0, n_feat - w, step):
    #                 temp_corr_arr.append(np.corrcoef(df.iloc[:, ind_i][i:i + w], df.iloc[:, ind_j][i:i + w])[
    #                                          0, 1])  # the covar matrix of 2x2
    #             n_wk = len([l for l in temp_corr_arr if np.abs(l) <= 0.25]) / len(temp_corr_arr)  # no corr %
    #             n_st = len([l for l in temp_corr_arr if np.abs(l) >= 0.8]) / len(temp_corr_arr)  # strong corr %
    #             arr_key = '__'.join([df.columns[ind_i], df.columns[ind_j]])
    #             # print (arr_key, n_wk, n_st)
    #             subjects_corr_df.loc[subj, arr_key + '_wk'] = n_wk
    #             subjects_corr_df.loc[subj, arr_key + '_st'] = n_st
    subjects_corr_df.to_csv(os.path.join(output_path , 'dFC.csv'), index=True)
    return subjects_corr_df


def main(flrd):
    # in_folders = [main_path + o + s for o in filt_pth for s in atl_pth]
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(create_dFC, in_folders)
    create_dFC(flrd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pass main subject folder")
    parser.add_argument('-i', type=str, required=True)
    args = parser.parse_args()
    main(args.i)