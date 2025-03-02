import numpy as np
import nibabel as nib
import torch.nn as nn
import torch
import torch.nn.functional as F
import data_preprocess
from tqdm import tqdm
from skimage.measure import marching_cubes
from scipy.spatial import distance


def crop_tensor(tensor):
    # Get the indices of non-zero (1) elements in each dimension
    non_zero_indices = torch.nonzero(tensor)

    if non_zero_indices.numel() == 0:
        # If there are no 1s, return the original tensor
        return tensor

    # Calculate the range to crop
    min_indices = non_zero_indices.min(dim=0)[0]
    max_indices = non_zero_indices.max(dim=0)[0]

    return min_indices, max_indices

def crop_with_same_indices(tensor_a, tensor_b, tensor_c):
    # Get the crop indices for tensor_a
    min_indices, max_indices = crop_tensor(tensor_a)

    # Crop tensor_a
    cropped_a = tensor_a[min_indices[0]:max_indices[0]+1,
                          min_indices[1]:max_indices[1]+1,
                          min_indices[2]:max_indices[2]+1]

    # Crop tensor_b and tensor_c using the same indices
    cropped_b = tensor_b[min_indices[0]:max_indices[0]+1,
                          min_indices[1]:max_indices[1]+1,
                          min_indices[2]:max_indices[2]+1]

    cropped_c = tensor_c[min_indices[0]:max_indices[0]+1,
                          min_indices[1]:max_indices[1]+1,
                          min_indices[2]:max_indices[2]+1]

    return cropped_a, cropped_b, cropped_c


def load_nii(file_path):
    """Load nii.gz file and return it as a numpy array."""
    nii = nib.load(file_path)
    return np.array(nii.get_fdata(), dtype=np.float32)


def precision_index(imageA, imageB):
    # Ensure the input images are binary images
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"
    
    # Calculate intersection and union
    intersection = np.sum((imageA == 1) & (imageB == 1))
    union = np.sum((imageA == 1) | (imageB == 1))
    
    # Calculate precision index
    if union == 0:
        return 1.0  # If both sets are empty, define the precision index as 1.0
    return intersection / union


def compute_precision_recall(pred, target):
    """
    Calculate Precision and Recall between predicted and target binary masks.

    Parameters:
    pred (numpy.ndarray): Predicted binary mask (0 or 1).
    target (numpy.ndarray): Ground truth binary mask (0 or 1).

    Returns:
    tuple: (precision, recall)
    """
    # Ensure the input arrays have the same shape
    assert pred.shape == target.shape, "Predicted and target arrays must have the same dimensions."

    # Calculate True Positives, False Positives, and False Negatives
    TP = torch.sum((pred == 1) & (target == 1))
    FP = torch.sum((pred == 1) & (target == 0))
    FN = torch.sum((pred == 0) & (target == 1))

    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return precision, recall



def compute_metrics(pred_path, target_path, input_path):
    """Calculate MSE, L1 Loss, Dice, Hausdorff Distance, mIoU, precision, and Recall for two nii.gz files."""
    pred = load_nii(pred_path)
    target = load_nii(target_path)
    input = load_nii(target_path)
    
    # Convert arrays to PyTorch tensors
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)
    input_tensor = torch.tensor(target, dtype=torch.float32)
    
    crop_tensor =  ((target_tensor+input_tensor) > 0.5).float()
    pred_tensor = (pred_tensor >= 0.5).float()
    target_tensor = (target_tensor >= 0.5).float()
    
    crop_tensor, pred_tensor, target_tensor = crop_with_same_indices(crop_tensor, pred_tensor, target_tensor)
    # MSE
    mse = nn.MSELoss()(pred_tensor, target_tensor).item()

    
    # Dice Score
    intersection = (pred_tensor * target_tensor).sum()
    dice = (2 * intersection) / (pred_tensor.sum() + target_tensor.sum()).item()
       
    # mIoU (mean Intersection over Union)
    intersection = (pred_tensor * target_tensor).sum()
    union = (pred_tensor + target_tensor).clamp(0, 1).sum()
    miou = (intersection / union).item()
    
    # precision and Recall
    precision, recall = compute_precision_recall(pred_tensor, target_tensor)

    return {
        "MSE": mse,
        "Dice Score": dice,
        "mIoU": miou,
        "precision": precision,
        "Recall": recall
    }


def compare_nifti_surfaces(file1_path, file2_path):
    # Load a NIfTI file and return its data
    def load_nifti(file_path):
        img = nib.load(file_path)
        return img.get_fdata()

    # Extract the surface using the marching cubes algorithm
    def extract_surface(data):
        verts, faces, _, _ = marching_cubes(data, level=0.5)  # 0.5 is used for binary (0-1) data
        return verts

    # Calculate the surface distance
    def surface_distance(verts1, verts2):
        dist1 = distance.cdist(verts1, verts2)
        min_dist1 = np.min(dist1, axis=1)  # Minimum distance from surface 1 to surface 2
        dist2 = distance.cdist(verts2, verts1)
        min_dist2 = np.min(dist2, axis=1)  # Minimum distance from surface 2 to surface 1
        
        return np.mean(min_dist1), np.mean(min_dist2)

    # Calculate HD95 (95th percentile of the Hausdorff distance)
    def hausdorff_distance(verts1, verts2):
        d1 = distance.cdist(verts1, verts2)
        hd1 = np.percentile(np.min(d1, axis=1), 95)
        
        d2 = distance.cdist(verts2, verts1)
        hd2 = np.percentile(np.min(d2, axis=1), 95)
        
        return hd1, hd2

    # Read NIfTI file data
    data1 = load_nifti(file1_path)
    data2 = load_nifti(file2_path)

    # Extract surfaces
    verts1 = extract_surface(data1)
    verts2 = extract_surface(data2)

    # Calculate surface distances
    mean_dist1, mean_dist2 = surface_distance(verts1, verts2)
    # print(f"Mean Surface Distance from model 1 to model 2: {mean_dist1:.4f}")
    # print(f"Mean Surface Distance from model 2 to model 1: {mean_dist2:.4f}")

    # Calculate HD95
    hd95_1, hd95_2 = hausdorff_distance(verts1, verts2)
    return hd95_1
    # print(f"HD95 from model 1 to model 2: {hd95_1:.4f}")
    # print(f"HD95 from model 2 to model 1: {hd95_2:.4f}")



def compute_mean_variance(values):
    """Compute the mean and variance of a 1D list."""
    mean_val = np.mean(values)
    variance_val = np.var(values)
    return mean_val, variance_val


if __name__ == '__main__':
    crop_size = (216, 96, 96)
    file_names = data_preprocess.get_file_names('./exp0226/MsDANet_1_bone_hole','.gz')
    mse, dice, miou, precision, recall, hd95 = [], [], [], [], [], []
    for _, file_name in tqdm(enumerate(file_names, 0), total=len(file_names) - 1, desc="\033[31mMeasuring:\033[0m"):
        if 'y_pre' in file_name:
            metrics = compute_metrics(file_name, file_name.replace('y_pre', 'y'), file_name.replace('y_pre', 'x'))
            tmp_hd95 = compare_nifti_surfaces(file_name, file_name.replace('y_pre', 'y'))
            mse.append(metrics["MSE"])
            dice.append(metrics["Dice Score"])
            miou.append(metrics["mIoU"])
            precision.append(metrics["precision"])
            recall.append(metrics["Recall"])
            hd95.append(tmp_hd95)
    print('MSE: ', compute_mean_variance(np.array(mse)))
    print('Dice Score: ', compute_mean_variance(np.array(dice)))
    print('mIoU: ', compute_mean_variance(np.array(miou)))
    print('precision: ', compute_mean_variance(np.array(precision)))
    print('Recall: ', compute_mean_variance(np.array(recall)))
    print('hd95: ', compute_mean_variance(np.array(hd95)))


            
