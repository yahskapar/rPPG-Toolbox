"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from tqdm import tqdm


def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []

    # No Motion
    # tasks = ["T2", "T8"]

    # Small Motion
    # tasks = ["T3", "T9"]

    # Medium Motion
    # tasks = ["T4", "T10"]

    # Large Motion
    # tasks = ["T5", "T11"]

    # Random Motion
    # tasks = ["T6", "T12"]

    # All tasks
    # tasks = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]

    # For V2 of UBFC-PHYS loader (all 168 vids)
    # T1
    subselect_out_T1 = ["s3_T1", "s8_T1", "s9_T1", "s26_T1", "s28_T1", "s30_T1", "s31_T1", "s32_T1", "s33_T1", "s40_T1", "s52_T1", "s53_T1", "s54_T1", "s56_T1"]
    # T2
    subselect_out_T2 = ["s1_T2", "s4_T2", "s6_T2", "s8_T2", "s9_T2", "s11_T2", "s12_T2", "s13_T2", "s14_T2", "s19_T2", "s21_T2", "s22_T2", "s25_T2", "s26_T2", "s27_T2", "s28_T2", "s31_T2", "s32_T2", "s33_T2", "s35_T2", "s38_T2", "s39_T2", "s41_T2", "s42_T2", "s45_T2", "s47_T2", "s48_T2", "s52_T2", "s53_T2", "s55_T2"]
    # T3
    subselect_out_T3 = ["s5_T3", "s8_T3", "s9_T3", "s10_T3", "s13_T3", "s14_T3", "s17_T3", "s22_T3", "s25_T3", "s26_T3", "s28_T3", "s30_T3", "s32_T3", "s33_T3", "s35_T3", "s37_T3", "s40_T3", "s47_T3", "s48_T3", "s49_T3", "s50_T3", "s52_T3", "s53_T3"]

    subselect_out = subselect_out_T1 + subselect_out_T2 + subselect_out_T3
    print("{} subjects will be ignored!".format(len(subselect_out)))

    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        # For AFRL, first ignore tasks we don't care about
        # if not any(s in test_batch[2][0] for s in tasks):
        #     continue
        # print(test_batch[2][0])
        
        # For PURE
        # task_number = test_batch[2][0][-2::]
        # if task_number != "02":
        #     continue    # We only want to evaluate videos related to the speech task
        # print(test_batch[2][0])

        # For UBFC-PHYS filtering
        if test_batch[2][0] in subselect_out:
            continue

        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            # Temp for AFRL
            # data_input = data_input[:, :, :, 3:]
            if method_name == "POS":
                # BVP = POS_WANG(data_input[:, :, :, 3:], config.UNSUPERVISED.DATA.FS)
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr, pre_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                predict_hr_peak_all.append(pre_hr)
                gt_hr_peak_all.append(gt_hr)
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_fft_hr, pre_fft_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                   fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                predict_hr_fft_all.append(pre_fft_hr)
                gt_hr_fft_all.append(gt_fft_hr)
    print("Used Unsupervised Method: " + method_name)
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_PEAK))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(
                    np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_PEAK))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(
                    np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_PEAK))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson  (FFT Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Wrong Test Metric Type")
