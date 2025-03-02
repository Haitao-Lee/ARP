# coding = utf-8
import argparse
import os
import torch.nn as nn
import torch
import Loss
# from initialize import *


dir_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
parser = argparse.ArgumentParser()
# model 
parser.add_argument('--save_model', default='./checkpoints')

# training
parser.add_argument('--epochs', default=200, help=None)
parser.add_argument('--patience', default=100)
parser.add_argument('--delta', default=1e-6) 
parser.add_argument('--train_loss', default=Loss.CombinedLoss()) # nn.MSELoss()) # 
parser.add_argument('--valid_loss', default=nn.MSELoss()) # 
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--loss_eps', default=30)
parser.add_argument('--train_img_dir', default=dir_path + '/dataset/train/img',
                    help='images of training data')
parser.add_argument('--train_label_dir', default=dir_path + '/dataset/train/label',
                    help='labels of training data')
parser.add_argument('--val_img_dir', default=dir_path + '/dataset/val/img',
                    help='images of validation data')
parser.add_argument('--val_label_dir', default=dir_path + '/dataset/val/label',
                    help='labels of validation data')
parser.add_argument('--test_img_dir', default=dir_path + '/dataset/val/img',
                    help='images of validation data')
parser.add_argument('--test_label_dir', default=dir_path + '/dataset/val/label',
                    help='labels of validation data')
parser.add_argument('--output', default=dir_path + '/exp0226',
                    help='the path of the prediction results')
# hardware setting
parser.add_argument('--device', default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),)
args = parser.parse_args()


