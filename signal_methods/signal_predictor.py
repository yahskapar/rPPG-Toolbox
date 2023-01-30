"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from evaluation.post_process import *
from signal_methods.methods.CHROME_DEHAAN import *
from signal_methods.methods.GREEN import *
from signal_methods.methods.ICA_POH import *
from signal_methods.methods.LGI import *
from signal_methods.methods.PBV import *
from signal_methods.methods.POS_WANG import *
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib as plt


def signal_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["signal"] is None:
        raise ValueError("No data for signal method predicting")
    print("===Signal Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    sbar = tqdm(data_loader["signal"], ncols=80)

    # T1
    # subselect_out = ["s3", "s8", "s9", "s26", "s28", "s30", "s31", "s32", "s33", "s40", "s52", "s53", "s54", "s56"]
    # T2
    # subselect_out = ["s1", "s4", "s6", "s8", "s9", "s11", "s12", "s13", "s14", "s19", "s21", "s22", "s25", "s26", "s27", "s28", "s31", "s32", "s33", "s35", "s38", "s39", "s41", "s42", "s45", "s47", "s48", "s52", "s53", "s55"]
    # T3
    subselect_out = ["s5", "s8", "s9", "s10", "s13", "s14", "s17", "s22", "s25", "s26", "s28", "s30", "s32", "s33", "s35", "s37", "s40", "s47", "s48", "s49", "s50", "s52", "s53"]
    print("{} subjects will be ignored!".format(len(subselect_out)))

    for _, test_batch in enumerate(sbar):
        # print(test_batch[2][0])
        if test_batch[2][0] in subselect_out:
            print("Skipping subject!")
            continue

        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.SIGNAL.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.SIGNAL.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.SIGNAL.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("signal method name wrong!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr, pre_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                fs=config.SIGNAL.DATA.FS, hr_method='Peak')
                predict_hr_peak_all.append(pre_hr)
                gt_hr_peak_all.append(gt_hr)
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_fft_hr, pre_fft_hr = calculate_metric_per_video(BVP, labels_input, diff_flag=False,
                                                                   fs=config.SIGNAL.DATA.FS, hr_method='FFT')
                predict_hr_fft_all.append(pre_fft_hr)
                gt_hr_fft_all.append(gt_fft_hr)
    print("Used Signal Method: " + method_name)
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        for metric in config.SIGNAL.METRICS:
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
        for metric in config.SIGNAL.METRICS:
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
    # Save a Bland-Altman Plot of Test Results
    sm.graphics.mean_diff_plot(gt_hr_fft_all, predict_hr_fft_all)
    plt.pyplot.savefig('bland_altman_plot_T3_MFC_SS_SIGNAL.png')
