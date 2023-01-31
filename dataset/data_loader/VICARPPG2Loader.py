import glob
import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

import pandas as pd

class VICARPPG2Loader(BaseLoader):
    """The data loader for the VicarPPG-2 dataset."""

    def __init__(self, name, data_path, config_data):
        """
        TODO: Update the docstring.
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For Vicar-PPG2 dataset)."""
        self.target = 'mov'

        data_dirs = glob.glob(data_path + os.sep + "*")
        self.root = data_path
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        videos = os.listdir(os.path.join(data_path,"Videos"))
        for data_dir in videos:
            if self.target not in data_dir:
                continue
            index = data_dir.split('-')[0]
            subject = "subject_"+index
            dirs.append({"index": index, "path": os.path.join(data_path,"Videos",data_dir), "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        gt = os.path.join(os.path.join(self.root,"GroundTruth","PPG","Cleaned"))
        f_name  = "{}-{} PPG.csv".format(data_dirs[i]['index'],self.target)
        p = os.path.join(gt,f_name)
        ppg = pd.read_csv(p).Signal.values
        print(data_dirs[i]['path'])
        print(ppg.shape)
        frames = self.read_video(data_dirs[i]['path'])

        target_length = frames.shape[0]
        print(target_length)
        exit()
        frames_clips, ppg_clips = self.preprocess(frames, ppg, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)
