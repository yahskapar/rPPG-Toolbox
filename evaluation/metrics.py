import os
import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *


def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)
    return np.reshape(sort_data.cpu(), (-1))


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    result_dict = dict()

    # T1
    # subselect_out = ["s3", "s8", "s9", "s26", "s28", "s30", "s31", "s32", "s33", "s40", "s52", "s53", "s54", "s56"]
    # T2
    # subselect_out = ["s1", "s4", "s6", "s8", "s9", "s11", "s12", "s13", "s14", "s19", "s21", "s22", "s25", "s26", "s27", "s28", "s31", "s32", "s33", "s35", "s38", "s39", "s41", "s42", "s45", "s47", "s48", "s52", "s53", "s55"]
    # T3
    # subselect_out = ["s5", "s8", "s9", "s10", "s13", "s14", "s17", "s22", "s25", "s26", "s28", "s30", "s32", "s33", "s35", "s37", "s40", "s47", "s48", "s49", "s50", "s52", "s53"]
    # print("{} subjects will be ignored!".format(len(subselect_out)))

    # For V2 of UBFC-PHYS loader (all 168 vids)
    # # T1
    # subselect_out_T1 = ["s3_T1", "s8_T1", "s9_T1", "s26_T1", "s28_T1", "s30_T1", "s31_T1", "s32_T1", "s33_T1", "s40_T1", "s52_T1", "s53_T1", "s54_T1", "s56_T1"]
    # # T2
    # subselect_out_T2 = ["s1_T2", "s4_T2", "s6_T2", "s8_T2", "s9_T2", "s11_T2", "s12_T2", "s13_T2", "s14_T2", "s19_T2", "s21_T2", "s22_T2", "s25_T2", "s26_T2", "s27_T2", "s28_T2", "s31_T2", "s32_T2", "s33_T2", "s35_T2", "s38_T2", "s39_T2", "s41_T2", "s42_T2", "s45_T2", "s47_T2", "s48_T2", "s52_T2", "s53_T2", "s55_T2"]
    # # T3
    # subselect_out_T3 = ["s5_T3", "s8_T3", "s9_T3", "s10_T3", "s13_T3", "s14_T3", "s17_T3", "s22_T3", "s25_T3", "s26_T3", "s28_T3", "s30_T3", "s32_T3", "s33_T3", "s35_T3", "s37_T3", "s40_T3", "s47_T3", "s48_T3", "s49_T3", "s50_T3", "s52_T3", "s53_T3"]

    # subselect_out = subselect_out_T1 + subselect_out_T2 + subselect_out_T3
    # print("{} subjects will be ignored!".format(len(subselect_out)))

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
    tasks = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]

    for index in predictions.keys():

        # For AFRL, first ignore tasks we don't care about
        if not any(s in index for s in tasks):
            continue
        print(index)

        # For UBFC-PHYS filtering
        # if index in subselect_out:
        #     continue

        # For PURE
        # task_number = index[-2::]
        # if task_number != "02":
        #     continue    # We only want to evaluate videos related to the speech task

        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized" or config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "Normalized":
            diff_flag_test = True
        else:
            raise ValueError("Not supported label type in testing!")
        gt_hr_fft, pred_hr_fft, label_ppg, pred_ppg = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
        gt_hr_peak, pred_hr_peak, label_ppg_peak, pred_ppg_peak = calculate_metric_per_video(
            prediction, label, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        predict_hr_peak_all.append(pred_hr_peak)
        gt_hr_peak_all.append(gt_hr_peak)

        # Store into dict for PCA/t-SNE plots
        result_dict[index] = {
            "gt_hr_fft": gt_hr_fft,
            "pred_hr_fft": pred_hr_fft,
            "label_ppg": label_ppg,
            "pred_ppg": pred_ppg
        }

    filename = config.TRAIN.MODEL_FILE_NAME + "_result.npy"
    file_path = os.path.join("/playpen-nas-ssd/akshay/UNC_Google_Physio/rPPG-Toolbox/dicts_for_paper", filename)
    np.save(file_path, result_dict)

    predict_hr_peak_all = np.array(predict_hr_peak_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                print("FFT MAE (FFT Label):{0}".format(MAE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                print("Peak MAE (Peak Label):{0}".format(MAE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "RMSE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                print("FFT RMSE (FFT Label):{0}".format(RMSE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                print("PEAK RMSE (Peak Label):{0}".format(RMSE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "MAPE":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                print("FFT MAPE (FFT Label):{0}".format(MAPE_FFT))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                print("PEAK MAPE (Peak Label):{0}".format(MAPE_PEAK))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        elif metric == "Pearson":
            if config.INFERENCE.EVALUATION_METHOD == "FFT":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                print("FFT Pearson (FFT Label):{0}".format(Pearson_FFT[0][1]))
            elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                print("PEAK Pearson  (Peak Label):{0}".format(Pearson_PEAK[0][1]))
            else:
                raise ValueError("Your evaluation method is not supported yet! Support FFT and peak detection now ")

        else:
            raise ValueError("Wrong Test Metric Type")

    # # Save a Bland-Altman Plot of Test Results
    # sm.graphics.mean_diff_plot(gt_hr_fft_all, predict_hr_fft_all)
    # plt.pyplot.savefig('bland_altman_plot_T3_MFC_SS_MAUBFC.png')
