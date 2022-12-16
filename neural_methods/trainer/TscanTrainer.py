"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from vidaug import augmentors as va
import random
from dataset.data_loader.BaseLoader import BaseLoader
import operator

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability

# SCAMPS_PURE_UBFC_TSCAN_BASIC_RAW_horizen_flip_0.5
# seq = va.Sequential([
#     sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_rotate10
# seq = va.Sequential([
#     va.RandomRotate(degrees=10) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_rotate20
# seq = va.Sequential([
#     va.RandomRotate(degrees=20) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_SCAMPS_PURE_TSCAN_RAW_rotate20
# seq = va.Sequential([
#     va.RandomRotate(degrees=20) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_SCAMPS_PURE_TSCAN_RAW_rotate20
# seq = va.Sequential([
#     va.RandomRotate(degrees=42) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# # SCAMPS_PURE_UBFC_TSCAN_RAW_rotate40
# seq = va.Sequential([
#     va.RandomRotate(degrees=40) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_rotate60
# seq = va.Sequential([
#     va.RandomRotate(degrees=60) # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_translate
# seq = va.Sequential([
#     va.RandomTranslate() # randomly rotates the video with a degree randomly choosen from [-10, 10]
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_rotate10_horizon_flip_0.5
# seq = va.Sequential([
#     va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
#     sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
# ])

# SCAMPS_PURE_UBFC_TSCAN_RAW_rotate20_horizon_flip_0.5
seq = va.Sequential([
    va.RandomRotate(degrees=20), # randomly rotates the video with a degree randomly choosen from [-10, 10]
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

class TscanTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.best_epoch = 0

    def train(self, data_loader):
        """ TODO:Docstring"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        min_valid_loss = 1
        train_loss_plot = []
        valid_loss_plot = []
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)

            # tbar_obj = enumerate(tbar)
            # max_tbar_index, max_tbar_value = max(enumerate(tbar_obj), key=operator.itemgetter(1))
            # print(max_tbar_index)

            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape

                # # Start of modified code from augmentations branch
                # C = C*2
                # data_stack_list = []
                # for batch_idx in range(N):
                #     data_list = []
                #     for frame_idx in range(D):
                #         data_numpy = data.detach()[batch_idx,frame_idx].cpu().permute(1,2,0).numpy()/255
                        
                        
                # #         # add video awgn noise

                # #         data_mean = np.mean(data_numpy)
                # #         data_noise_amp = np.random.randn(1) * data_mean * 0.05
                # #         data_numpy = data_numpy + np.random.randn(72,72,3) * data_noise_amp
                        

                #         data_list.append(data_numpy)
                #     video_aug = data_list

                # #     # add transformation 
                    
                #     video_aug = seq(data_list)
                # #     # plt.imshow(video_aug[0])
                # #     # plt.show()
                    
                #     diff_normalize_data_part = self.diff_normalize_data(video_aug)
                #     standardized_data_part = self.standardized_data(video_aug)
                #     cat_data = np.concatenate((diff_normalize_data_part,standardized_data_part),axis=3)
                #     data_stack_list.append(cat_data)
                # data_stack = np.asarray(data_stack_list)
                # data_stack_tensor = torch.zeros([N, D, C, H, W], dtype = torch.float)
                # for batch_idx in range(N):
                #     data_stack_tensor[batch_idx] = torch.from_numpy(data_stack[batch_idx]).permute(0,3,1,2).to(self.device)
                # data = data_stack_tensor

                # # add label awgn noise
 
                # # if random.random() > 0.5:
                # #     labels = labels + (0.75 ** 0.5) * torch.randn(4, 180).to(self.device)
                

                # # End of modified code from augmentations branch

                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                # train_loss_plot.insert(((epoch * max_tbar_index + 1) + idx), loss.item())
                tbar.set_postfix(loss=loss.item())
            # np.save('train_loss.npy', np.array(train_loss))
            train_loss_plot.insert((epoch + 1), np.mean(train_loss))
            valid_loss, valid_loss_plot = self.valid(data_loader, epoch, valid_loss_plot)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            valid_loss_plot.insert((epoch + 1), valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        np.save('train_loss_plot_epoch.npy', np.array(train_loss_plot))
        np.save('valid_loss_plot_epoch.npy', np.array(valid_loss_plot))
        print("best trained epoch:{}, min_val_loss:{}".format(self.best_epoch, min_valid_loss))

    def valid(self, data_loader, epoch, valid_loss_plot):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)

            # vbar_obj = enumerate(vbar)
            # max_vbar_index, max_vbar_value = max(enumerate(vbar_obj), key=operator.itemgetter(1))
            # print(max_vbar_index)

            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                # valid_loss_plot.insert(((epoch * max_vbar_index + 1) + valid_idx), loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
            # np.save('valid_loss.npy', valid_loss)
        return np.mean(valid_loss), valid_loss_plot

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    # Practically the same thing as what's in BaseLoader.py
    def diff_normalize_data(self, data):
        """Difference frames and normalization data"""
        normalized_len = len(data)
        h, w, c = data[0].shape
        normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
        normalized_data[normalized_len-1] = (data[normalized_len-1] - data[normalized_len-2]) / (
                    data[normalized_len-1] + data[normalized_len-2] + 1e-7)
        for j in range(normalized_len - 1):
            normalized_data[j] = (data[j + 1] - data[j]) / (
                    data[j + 1] + data[j] + 1e-7)
        normalized_data = normalized_data / np.std(normalized_data)
        normalized_data[np.isnan(normalized_data)] = 0
        return normalized_data

    # Practically the same thing as what's in BaseLoader.py
    def standardized_data(self, data):
        """Difference frames and normalization data"""
        data = np.asarray(data)
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data
