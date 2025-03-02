# coding = utf-8
from initialize import *
from config import args
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import data_set
import torch.optim as optim
import models.UViT
# CNN models
import models.UNet_3plus
import models.ConvNext
import models.nnunet
import models.sctnet
import models.FAMNet
import models.ConDSeg
import models.FocalNet

# Transformer models
import models.swin_transformer
import models.transunet
import models.ITPN
import models.ViT
import models.unetr_pp

# Mamba models
import models.VisionMamba
import models.UMamba
import models.SparX
import models.segmamba
import models.MobileMamba

# Ours
import models.MsDANet

import os
import logging
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Get the absolute path of the log file
log_path = os.path.abspath('./models/MsDANet_1_bone_hole.log')
print("Log path:", log_path)

# Create the log directory if it does not exist
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level to INFO to include informational messages

# Check if a FileHandler has already been added to avoid duplication
if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
    file_handler = logging.FileHandler(log_path, mode='a')  # Open the log file in append mode
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))  # Set log format
    logger.addHandler(file_handler)  # Add the FileHandler to the logger


class early_stop():
    def __init__(self, patience=args.patience, verbose=False, delta=args.delta):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), path + '/' + 'MsDANet_1_bone_hole.pth')
        self.val_loss_min = val_loss


def main():
    device = args.device
    # loss_eps = args.loss_eps
    train_loader = data_set.train_loader
    valid_loader = data_set.valid_loader
    print(device)
    
    # CNN models
    # net = models.ConvNext.ConvNeXtV2().to(device) 
    # net = models.sctnet.SCTNet().to(device)
    # net = models.FocalNet.FocalNet().to(device)
    # net = models.ConDSeg.ConDSeg().to(device)
    # net = models.nnunet.Generic_UNet().to(device)
        
    # Transformer models
    # net = models.transunet.TransUNet().to(device)
    # net = models.swin_transformer.SwinTransformerV2().to(device)
    # net = models.ITPN.iTPN().to(device)
    # net = models.ViT.VisionTransformer().to(device)
    # net = models.unetr_pp.UNETR_PP().to(device)
    
    # Mamba models
    # net = models.VisionMamba.VisionMamba().to(device)
    # net = models.UMamba.UMamba().to(device)
    # net = models.segmamba.SegMamba().to(device)
    # net = models.SparX.sparx_mamba_b().to(device)
    
    
    # ours
    net = models.MsDANet.MultiscaleDenseAttentionNet(down_scale=1, device=device).to(device)
    # state_dict = torch.load(args.save_model + '/MsDANet_3.pth')
    # net.load_state_dict(state_dict, strict=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion_train = args.train_loss
    criterion_valid = args.valid_loss
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    # dist_flag = True
    early_stopping = early_stop(patience=args.patience, verbose=True)
    # =======================train=====================    ===
    # scaler = torch.amp.GradScaler("cuda")
    # with torch.amp.autocast('cuda'):
    for epoch in range(args.epochs):
        # if epoch % 10 == 0:
        #     print('\033[31mtraining:\033[0m'.format(epoch + 1))
        logging.info(f"Starting epoch {epoch}:")
        net.train()
        train_epoch_loss = []
        train_tmp_loss = []
        for idx, (inputs, targets) in tqdm(enumerate(train_loader, 0), total=len(train_loader) - 1, desc="\033[31mEpoch {}, training:\033[0m".format(epoch + 1)):
            # targets_tmp = targets
            inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
            outputs = net(inputs)[0]
            optimizer.zero_grad()
            loss = criterion_train(outputs, targets, inputs)   # criterion_train(outputs, targets)  # 
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_tmp_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx == len(train_loader) - 1:
                print("\nepoch={}/{}, loss={}".format(epoch, args.epochs, np.average(train_tmp_loss)))
                logging.info("epoch={}/{}, loss={}".format(epoch, args.epochs, np.average(train_tmp_loss)))
    
        train_epochs_loss.append(np.average(np.average(train_tmp_loss)))
        # =====================valid============================
        net.train()
        valid_epoch_loss = []
        with torch.no_grad():
            for idx, (inputs, targets) in tqdm(enumerate(valid_loader), total=len(valid_loader),desc="\033[31mEpoch {}, validation:\033[0m".format(epoch + 1)):
                torch.cuda.empty_cache()
                inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                outputs = net(inputs)[0]
                outputs = (outputs >= 0.5).float()
                targets = (targets >= 0.5).float()
                loss = criterion_valid(outputs, targets)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        # ==================early stopping======================
        early_stopping(valid_epochs_loss[-1], model=net, path=args.save_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
            # ====================adjust lr========================
            # lr_adjust = {
            #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            #     10: 5e-7, 15: 1e-7, 20: 5e-8
            # }
            # if epoch in lr_adjust.keys():
            #     lr = lr_adjust[epoch]
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            #     print('Updating learning rate to {}'.format(lr))
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    train_epochs_loss = np.array(train_epochs_loss)
    valid_epochs_loss = np.array(valid_epochs_loss)
    np.save('./loss/train_loss_unet.npy',train_loss)
    np.save('./loss/valid_loss_unet.npy',valid_loss)
    np.save('./loss/train_epochs_loss_unet.npy',train_epochs_loss)
    np.save('./loss/valid_epochs_loss_unet.npy',valid_epochs_loss)
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[1:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('./Loss.png')
    plt.show()


if __name__ == '__main__':
    main()
