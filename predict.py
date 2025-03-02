# coding = utf-8

from config import args
import models.ConvNext
import models.UNet_3plus
import models.densenet
import os
import random
import numpy as np
import matplotlib.pyplot as plt
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
import copy
from tqdm import tqdm
import data_preprocess
import SimpleITK as sitk
import models.MsDANet
import torch
import torch.nn.functional as F
import vtk
import Loss
from vtk.util import numpy_support



def smooth_nifti(input_image, output_nifti_path, sigma=2.5):
    """
    Smooth a NIfTI image using VTK's Gaussian smoothing filter.

    Parameters:
    - input_nifti: image,.
    - output_nifti_path: str, path to save the smoothed NIfTI file.
    - sigma: float, standard deviation for the Gaussian smoothing.
    """
    # Convert SimpleITK image to numpy array
    input_array = sitk.GetArrayFromImage(input_image)

    # Convert numpy array to VTK image
    vtk_image = vtk.vtkImageImport()
    vtk_image.SetWholeExtent(0, input_array.shape[2]-1, 0, input_array.shape[1]-1, 0, input_array.shape[0]-1)
    vtk_image.SetDataExtent(0, input_array.shape[2]-1, 0, input_array.shape[1]-1, 0, input_array.shape[0]-1)
    vtk_image.CopyImportVoidPointer(input_array, input_array.nbytes)
    vtk_image.SetDataScalarTypeToFloat()  # Adjust if your data type is different
    vtk_image.SetNumberOfScalarComponents(1)
    vtk_image.Update()

    # Step 2: Apply Gaussian smoothing filter
    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputConnection(vtk_image.GetOutputPort())
    smoother.SetStandardDeviations(sigma, sigma, sigma)  # Adjust for smoothing level
    smoother.Update()

    # Step 3: Convert smoothed VTK image back to numpy array
    smoothed_vtk_image = smoother.GetOutput()
    smoothed_array = numpy_support.vtk_to_numpy(smoothed_vtk_image.GetPointData().GetScalars())
    smoothed_array = smoothed_array.reshape(input_array.shape)
    smoothed_array[smoothed_array>=0.5] = 1
    smoothed_array[smoothed_array<0.5] = 0

    # Step 4: Convert numpy array back to SimpleITK image
    smoothed_image = sitk.GetImageFromArray(smoothed_array)
    smoothed_image.CopyInformation(input_image)  # Copy metadata (spacing, origin, etc.)

    # Step 5: Save the smoothed image using SimpleITK
    sitk.WriteImage(smoothed_image, output_nifti_path)


def predict():
    device = args.device
    
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
    
    net = models.MsDANet.MultiscaleDenseAttentionNet(down_scale=1, device=device).to(device)
    state_dict = torch.load(args.save_model +'/' + 'MsDANet_1_bone_hole.pth')
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    file_path_list = os.listdir(args.test_img_dir)
    idx = 0
    aver_loss = []
    with torch.no_grad():
        for idx, path in tqdm(enumerate(file_path_list, 0), total=len(file_path_list) - 1, desc="\033[31mTesting Data:\033[0m"):
            x_image = sitk.ReadImage(args.val_img_dir + '/' + path)
            y_image = sitk.ReadImage(args.val_label_dir + '/' + path)
            x_np = data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(x_image))
            y_np = data_preprocess.pad_image_to_size(sitk.GetArrayFromImage(y_image))
            x_np[x_np < 0.5] = -1
            x_np[x_np >= 0.5] = 1
            y_np[y_np < 0.5] = -1
            y_np[y_np >= 0.5] = 1
            x_ts, y_ts = torch.from_numpy(x_np[None, ]).to(torch.float32).to(device), torch.from_numpy(y_np[None, ]).to(torch.float32).to(device)
            # a = net(x_ts)
            y_pre_ts = net(x_ts)[0] + (x_ts >= 0.5).float()
            y_pre_ts = (y_pre_ts >= 0.5).float()
            y_ts = (y_ts >= 0.5).float()
            y_pre_ts[0, :, :, :] = Loss.keep_largest_cluster(y_pre_ts[0, :, :, :])
            # print(y_ts.shape)
            # print("Unique elements in the tensor:", torch.unique(y_ts))
            # print(torch.sum(y_ts))
            # print(y_pre_ts.shape)
            # print(y_pre_ts.shape)
            loss = args.valid_loss(y_pre_ts[0, :, :, :], y_ts[0, :, :, :]).item()
            aver_loss.append(loss)
            y_pre_np = y_pre_ts.cpu().detach().numpy()[0, :, :, :]
            output = args.output + f'/MsDANet_1_bone_hole/{idx}_mseloss_{loss}'
            if not os.path.exists(output):
                os.makedirs(output)
            y_pre_np[y_pre_np < 0.5] = 0
            y_pre_np[y_pre_np >= 0.5] = 1
            # print(np.unique(y_pre_np))
            # print(np.sum(y_pre_np))
            # print(y_pre_np.shape)
            y_pre_image = sitk.GetImageFromArray(y_pre_np)
            y_pre_image.SetSpacing(x_image.GetSpacing())
            y_pre_image.SetOrigin(x_image.GetOrigin())
            y_pre_image.SetDirection(x_image.GetDirection())
            # print(sitk.GetArrayFromImage(y_pre_image).shape)
            y_image = sitk.GetImageFromArray(y_np)
            y_image.SetSpacing(x_image.GetSpacing())
            y_image.SetOrigin(x_image.GetOrigin())
            y_image.SetDirection(x_image.GetDirection())
            sitk.WriteImage(x_image, output + '/x.nii.gz')
            sitk.WriteImage(y_image, output + '/y.nii.gz')
            smooth_nifti(y_pre_image, output + '/y_pre.nii.gz')
            # sitk.WriteImage(y_pre_image, output + '/y_pre.nii.gz')
    # predict_x_dir = []
    # for path in os.listdir(args.exp_train_lut):
    #     predict_x_dir.append(args.exp_train_lut +'/'+ path)
    # predict_y_dir = []
    # for path in os.listdir(args.exp_train_peak):
    #     predict_y_dir.append(args.exp_train_peak +'/'+ path)
    # val_idx = random.sample(range(len(predict_x_dir)), 10)
    # net.eval()
    # for idx in val_idx:
    #     lut = np.load(predict_x_dir[idx]) #[256,256]
    #     tmp_lut = (lut - lut.min())/(lut.max()-lut.min() + 1e-4)
    #     input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [-1,1,256,256]
    #     peak = np.load(predict_y_dir[idx]) # [256,2] 
    #     tmp_peak = peak
    #     input = torch.from_numpy(input)
    #     input = input.to(torch.float32).to(device)
    #     predict_peak = net(input)
    #     #loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2) - data_set.mean_cordinate).to(torch.float32).to(device))
    #     loss = args.valid_loss(predict_peak, (torch.flatten(torch.from_numpy(np.repeat(tmp_peak[None,None,...], 1, axis=0)),2)).to(torch.float32).to(device))
    #     print('loss:',loss.item()  )
    #     predict_peak = predict_peak.cpu().detach().numpy() #+ data_set.mean_cordinate
    #     predict_peak = predict_peak[0,0,:]
    #     predict_peak.resize([256,2]) #[256,2]
    #     predict_peak = np.round(predict_peak).astype(int)
    #     mean = copy.deepcopy(data_set.mean_cordinate)
    #     mean.resize([256,2])
    #     mean = np.round(mean).astype(int)
    #     a = np.linalg.norm(predict_peak - peak, axis=1)
    #     # tmp = np.abs(predict_peak - peak)
    #     print("max offset:", np.linalg.norm(predict_peak - peak, axis=1).max())
    #     # print(predict_peak.shape)
    #     _, axes = plt.subplots(nrows=1, ncols=3)
    #     # predict_peak_lut = np.zeros(lut.shape)
    #     # for j in range(predict_peak.shape[0]):
    #     #     predict_peak_lut[min(max(0, predict_peak[j][0]), 255), min(max(0, predict_peak[j][1]), 255)] = 1
    #     # origin_peak_lut = np.zeros(lut.shape)
    #     # for j in range(peak.shape[0]):
    #     #     origin_peak_lut[min(max(0, peak[j][0]), 255), min(max(0, peak[j][1]), 255)] = 1
    #     compare_lut = np.zeros(lut.shape)
    #     # for j in range(peak.shape[0]):
    #     #     compare_lut[min(max(0, peak[j][0]), 255), min(max(0, peak[j][1]), 255)] = 2
    #     #     compare_lut[min(max(0, predict_peak[j][0]), 255), min(max(0, predict_peak[j][1]), 255)] = 0
    #     axes[0].matshow(lut,  alpha=1)
    #     axes[0].plot(predict_peak[:, 1],predict_peak[:, 0], "ro")
    #     axes[0].set_title('predict%d' % idx)
    #     axes[1].matshow(lut, alpha=1)
    #     axes[1].plot(peak[:, 1],peak[:, 0], "b*")
    #     axes[1].set_title('origin%d' % idx)
    #     axes[2].matshow(lut,  alpha=1)
    #     # axes[2].matshow(compare_lut, alpha=0.5)
    #     axes[2].plot(predict_peak[:, 1],predict_peak[:, 0], "ro")
    #     axes[2].plot(peak[:, 1],peak[:, 0], "b*")
    #     axes[2].plot(mean[:, 1],mean[:, 0], "gx")
    #     axes[2].set_title('compare%d' % idx)
    #     # plt.savefig(args.predict_save + '/data%d.png' % idx)
    #     plt.show()
    print(np.average(aver_loss))
    return 0


if __name__ == '__main__':
    predict()
        
        
        
    
    
    