# coding = utf-8
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from config import args     
import SimpleITK as sitk   
from tqdm import tqdm
import data_preprocess


class dataset(Dataset):
    def __init__(self, xs, ys, flag='training'):
        assert flag in ['training', 'validation']
        self.flag = flag
        self.xs = xs
        self.ys = ys

    def __getitem__(self, index):
        assert index in range(0, len(self.xs))
        _x = self.xs[index] # shape=(128,128,128)
        _y = self.ys[index] # shape=(128,128,128)
        _x[_x < 0.5] = -1
        _x[_x >= 0.5] = 1
        _y[_y < 0.5] = 0
        _y[_y >= 0.5] = 1
        return _x, _y
    
    def __len__(self): 
        return len(self.xs)
    
    @staticmethod
    def normalize_01(x):
        _m = x.min()
        return (x - _m) / (x.max() - _m + 1e-4)
    
    @staticmethod
    def normalize_255(x):
        return x/255    
              

train_xs = []
train_ys = []
file_path_list = os.listdir(args.train_img_dir)
for _, path in tqdm(enumerate(file_path_list, 0), total=len(file_path_list) - 1, desc="\033[31mCollecting Training Data:\033[0m"):
    train_xs.append(data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(sitk.ReadImage(args.train_img_dir + '/' + path))))
    train_ys.append(data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(sitk.ReadImage(args.train_label_dir + '/' + path))))
valid_xs = []
valid_ys = []
file_path_list = os.listdir(args.val_img_dir)
for _, path in tqdm(enumerate(file_path_list, 0), total=len(file_path_list) - 1, desc="\033[31mCollecting Validation Data:\033[0m"):
    valid_xs.append(data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(sitk.ReadImage(args.val_img_dir + '/' + path))))
    valid_ys.append(data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(sitk.ReadImage(args.val_label_dir + '/' + path))))


train_dataset = dataset(flag='training', xs=train_xs, ys=train_ys)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
valid_dataset = dataset(flag='validation', xs=valid_xs, ys=valid_ys)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)
