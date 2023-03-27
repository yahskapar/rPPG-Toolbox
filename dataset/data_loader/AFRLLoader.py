"""The dataloader for AFRL datasets.

TO DO: Add dataset citation
"""
import os
import cv2
import glob
import json
import numpy as np
import re
from dataset.data_loader.BaseLoader import BaseLoader
# from utils.utils import sample
import glob
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import mat73
import signal

class AFRLLoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an AFRL dataloader.


        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""
        
        return data_path # returns base data path for use by `preprocess_dataset`
    
    def split_raw_data(self, data_dirs, begin, end):
        print(data_dirs)
        """Returns a subset of data dirs, split with begin and end values"""
        if begin == 0 and end == 1: # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i):
        """   invoked by preprocess_dataset for multi_process.   """
        file_info = data_dirs[i]
        path = file_info[0]
        fname = file_info[1]
        saved_filename = fname.replace('VideoB2', '')

        frames = self.read_video(path) # read in video data
        bvps = self.read_wave(path)

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, config_preprocess.LARGE_FACE_BOX)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):

        # get data path and files - this is not done in 'get_data' as we are working with pre-chunked data and not raw video files
        
        # get frame size to find right data dir
        if config_preprocess.H == 72 and config_preprocess.W == 72:
            data_path = os.path.join(data_dirs, "ProcessedInputFiles/AFRLChunks72x72")
        elif config_preprocess.H == 36 and config_preprocess.W == 36:
            data_path = os.path.join(data_dirs, "ProcessedInputFiles/AFRLChunks36x36SS")
        else:
            raise ValueError(self.name + ' Frame size in config file not supported for AFRL dataset (only 72x72 and 36x36 supported)')

        # Get all dataset participants
        # Mat files in format: PXXTXVideoB2CXX 
        # This translates to: Participant (P) XX, Trial (T) X, Chunk (C) XX

        # dict of mat files 
        data_info = dict()
        data_file_paths = glob.glob(data_path + os.sep + "*.mat")

        for path in data_file_paths:

            fname = os.path.basename(path)
            fname = fname.replace('.mat', '') # remove file extension

            # not sure what this file is... but it exists in the dataset...
            if fname == 'M':
                continue

            P_loc = fname.find('P') # find the location of the P char in file name
            T_loc = fname.find('T') # find the location of the T char in file name
            V_loc = fname.find('V') # find the location of the V char in file name
            C_loc = fname.find('C') # find the location of the C char in file name

            subj_num = int(fname[P_loc + 1:T_loc])
            trial_num = int(fname[T_loc + 1:V_loc])
            chunk_num = int(fname[C_loc + 1:])

            if subj_num not in data_info: # if subject not in the data info dictionary
                data_info[subj_num] = [] # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subj_num].append((path, fname, subj_num, trial_num, chunk_num))
        
        # Data exploration has confirmed that ALL subjects have the same number of clips / total video duration
        subj_list = list(data_info.keys()) # all subjects by number ID (1-27)
        subj_list.sort()
        num_subjs = len(subj_list) # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0,num_subjs))
        if (begin !=0 or end !=1):
            subj_range = list(range(int(begin*num_subjs), int(end*num_subjs)))
            print(subj_range)
        print('subject ids:', subj_list)

        # compile file list
        file_info_list = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            file_info_list += subj_files # add file information to file_list (tuple of fname, subj ID, trial num, chunk num)

        # set up pbar
        file_num = len(file_info_list)
        choose_range = range(0,file_num)
        pbar = tqdm(list(choose_range))
        
        # multi_process
        p_list = []
        running_num = 0

        for i in choose_range:
            process_flag = True
            while (process_flag):         # ensure that every i creates a process
                if running_num <8:       # in case of too many processes

                    # print('FILENAME', file_info_list[0] ) # TO DO GIRISH

                    p = Process(target=self.preprocess_dataset_subprocess, args=(file_info_list, config_preprocess, i))
                    p.start()
                    p_list.append(p)
                    running_num +=1
                    process_flag = False
                for p_ in p_list:
                    if (not p_.is_alive() ):
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        # append all data path and update the length of data
        inputs = glob.glob(os.path.join(self.cached_path, "*input*.npy"))
        if inputs == []:
            raise ValueError(self.name + ' dataset loading data error!')
        labels = [input.replace("input", "label") for input in inputs]
        assert (len(inputs) == len(labels))
        self.inputs = inputs
        self.labels = labels
        self.len = len(inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video mat file, returns frames(T,H,W,3) """
        
        # read in .mat file
        mat = mat73.loadmat(video_file) # load in mat data
        frames_data = mat['dXsub']
        # frames = frames_data[:,:,:,3:6] # first 3 color channels are normalized
        frames = frames_data

        # print('READ IN FRAME SHAPE: ', frames.shape) # TO DO GIRISH

        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""

        # read in .mat file
        mat = mat73.loadmat(bvp_file) # load in mat data
        dPPG = mat['dysub']
        #dPPG = np.cumsum(dPPG)
        dResp = mat['drsub']
        #dResp = np.cumsum(dResp)

        # GIRISH TO DO
        # DO I need to detrend these???

        return np.asarray(dPPG)

    # Overwrite BaseLoader
    def preprocess(self, frames, bvps, config_preprocess, large_box=False):

        frames_clips = np.array([frames])
        bvps_clips = np.array([bvps])

        print(frames_clips.shape)
        print(bvps_clips.shape)

        return frames_clips, bvps_clips
