# coding = utf-8
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label, find_objects
import SimpleITK as sitk
from sklearn.cluster import DBSCAN
import nibabel as nib
from skimage import measure
from scipy.spatial import Delaunay


def compute_surface_curvature(data):
    """
    Calculate the surface curvature for the given NIfTI data.

    Parameters:
    data: array - NIfTI file array of shape [216, 96, 96]

    Returns:
    curvatures: np.ndarray - Curvature at each point on the surface
    """
    # Convert the PyTorch tensor to a NumPy array if it's not already
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Extract surface
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)  # Extract surface using marching cubes

    # Compute normals
    tri = Delaunay(verts)  # Create Delaunay triangulation
    normals = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], 
                       verts[faces[:, 2]] - verts[faces[:, 0]])  # Calculate normals
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]  # Normalize normals

    # Compute curvature (mean curvature)
    curvatures = np.zeros(verts.shape[0])  # Initialize curvature array

    for i in range(len(verts)):
        # Find adjacent triangles for the current vertex
        adjacent_faces = np.where(tri.simplices == i)[0]
        if len(adjacent_faces) < 3:
            continue  # Skip if less than 3 adjacent triangles

        # Calculate curvature
        angles = []
        for face in adjacent_faces:
            # Get the indices of the triangle vertices
            triangle_indices = tri.simplices[face]

            # Ensure indices are valid
            if any(idx >= len(normals) for idx in triangle_indices):
                print(f"Warning: Index out of bounds in triangle indices: {triangle_indices}")  # Debug info
                continue  # Skip if any index is out of bounds
            
            # Get the corresponding normals
            face_normals = normals[face]  # This is a (3, 3) array

            # Calculate angle between the normal of the triangle and the normals of the vertex
            for fn in face_normals:
                angle = np.arccos(np.clip(np.dot(normals[i], fn), -1.0, 1.0))
                angles.append(angle)

        if angles:  # Check if angles were calculated
            curvatures[i] = np.mean(angles)  # Store the mean angle as curvature
        else:
            curvatures[i] = 0  # If no angles, set curvature to 0

    return curvatures  # Return the curvature values



def compute_surface_to_volume_ratio(data):
    """
    Calculate the Surface-to-Volume Ratio for the given NIfTI file.

    Parameters:
    nii_file: str - Path to the NIfTI file

    Returns:
    surface_to_volume_ratio: float - Surface-to-Volume Ratio
    volume: int - Volume
    surface_area: float - Surface Area
    """
    # Convert the PyTorch tensor to a NumPy array if it's not already
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    else:
        data = data
    # Calculate volume
    volume = np.sum(data)  # Count the number of voxels with value 1
    # Extract surface
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)  # Extract surface using marching cubes
    # Calculate surface area
    # Triangle area calculation: 0.5 * |AB x AC|
    def triangle_area(v0, v1, v2):
        return np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0  # Calculate area of a triangle
    surface_area = sum(triangle_area(verts[faces[i, 0]], verts[faces[i, 1]], verts[faces[i, 2]]) for i in range(faces.shape[0]))  # Sum areas of all triangles
    # Calculate surface-to-volume ratio
    surface_to_volume_ratio = surface_area / volume if volume > 0 else 0  # Calculate surface-to-volume ratio
    return surface_to_volume_ratio, volume, surface_area  # Return the results



def keep_largest_cluster(tensor, eps=1.5):
    # Convert the PyTorch tensor to a NumPy array if it's not already
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.cpu().numpy()
    else:
        tensor_np = tensor
    
    # Get the coordinates of all points with a value of 1
    coords = np.argwhere(tensor_np == 1)
    if coords.shape[0] == 0:
         return tensor
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps).fit_predict(coords)
    clus_uniq = np.unique(np.array(clustering))
    if len(clus_uniq) == 0:
         return tensor
    largest_cluster = None
    largest_shape = 0
    # print(clus_uniq)
    for clus in clus_uniq:
        indices = np.argwhere(clustering == clus).flatten()
        ps = np.array(coords[indices])
        if ps.shape[0] > largest_shape:
            largest_shape = ps.shape[0]
            largest_cluster = ps
    
    # Create a new array of the same shape, filled with zeros
    largest_cluster_tensor = np.zeros_like(tensor_np)
    
    # Set the coordinates in the largest cluster to 1
    largest_cluster_tensor[tuple(largest_cluster.T)] = 1
    
    return torch.tensor(largest_cluster_tensor, dtype=tensor.dtype, device=tensor.device)


def remove_small_components(image):
    # Convert the image to binary (assuming background is 0, foreground is 1)
    binary_image = sitk.BinaryThreshold(image, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)

    # Connected component analysis
    cc = sitk.ConnectedComponent(binary_image)
    
    # Calculate the size of each connected component and retain the largest one
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    
    # Generate a binary image containing only the largest connected component
    largest_component = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)

    # Save the result
    return largest_component



def keep_largest_connected_component(tensor):
    # Convert to NumPy array
    array = tensor.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    
    # Perform connected component labeling
    labeled_array, num_features = label(array)
    
    if num_features == 0:
        return tensor  # Return original tensor if there are no connected components
    
    # Find the size of each connected component
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Ignore background
    
    # Get the label of the largest connected component
    largest_component_label = sizes.argmax()
    
    # Create a new array to keep the largest connected component
    largest_component = np.zeros_like(array)
    largest_component[labeled_array == largest_component_label] = 1
    
    # Convert back to PyTorch tensor and return
    return torch.tensor(largest_component).unsqueeze(0).to(tensor.device)


def morphological_operations(pred, kernel_size=3, operation="dilation"):
    """
    Apply 3D morphological operations (dilation or erosion) to smooth binary predictions.

    Args:
        pred (torch.Tensor): Binary input tensor with shape (B, D, H, W) and values 0 and 1.
        kernel_size (int): Size of the 3D structuring element (should be odd).
        operation (str): Either 'dilation' or 'erosion'.
        
    Returns:
        torch.Tensor: Smoothed binary tensor with shape (B, D, H, W).
    """
    # Ensure the input is a float tensor
    pred = pred.float()  # Convert binary input to float

    # Create a 3D structuring element (kernel) for morphological operations
    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=pred.device)

    if operation == "dilation":
        # Perform dilation to expand regions of 1s
        dilated = F.conv3d(pred, kernel, padding=kernel_size//2)
        # Threshold to get binary output
        return (dilated > 0).float()  # Keep as float for consistency
    elif operation == "erosion":
        # Perform erosion to shrink regions of 1s
        eroded = F.conv3d(pred, kernel, padding=kernel_size//2)
        # Threshold to get binary output
        return (eroded == kernel.numel()).float()  # Only return regions that are fully 1s
    else:
        raise ValueError("Invalid operation type. Use 'dilation' or 'erosion'.")


def smooth_predictions(pred, kernel_size=7):
    """
    Smooth binary predictions using morphological operations.
    
    Args:
        pred (torch.Tensor): Binary input tensor with shape (B, D, H, W) and values 0 and 1.
        kernel_size (int): Size of the kernel for morphological operations.
        
    Returns:
        torch.Tensor: Smoothed binary tensor with shape (B, D, H, W).
    """
    # Apply dilation followed by erosion
    dilated_pred = morphological_operations(pred, kernel_size, operation="dilation")
    # Then apply erosion
    smoothed_pred = morphological_operations(dilated_pred, kernel_size, operation="erosion")
    return smoothed_pred



class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1, smooth_weight=1, bmse_weight=1, hmse_weight=1, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.dice_loss = DiceLoss(smooth=smooth)  
        self.mse_weight = mse_weight
        self.bmse_weight = bmse_weight
        self.hmse_weight = hmse_weight
        self.smooth_weight = smooth_weight

    def forward(self, y_pred, y_true, x):
        mse = self.mse_loss(y_pred, y_true)
        y1 = (y_true >= 0.5).float()
        x1 = (x >= 0.5).float()
        hole_weight_mtx = y1-x1
        hole_weight_mtx = morphological_operations((hole_weight_mtx >= 0.5).float(), 15, 'dilation')
        hole_weight_mtx = (hole_weight_mtx >= 0.5).float()
        bone_weight_mtx = ((x1 + y1) >= 0.5).float()
        hole_mse= self.mse_loss(hole_weight_mtx*y_pred, hole_weight_mtx*y_true)
        bone_mse= self.mse_loss(bone_weight_mtx*y_pred, bone_weight_mtx*y_true)
        # smooth_punish = compute_surface_curvature((y_pred >= 0).float()[0, :, :, :])
        #dice = self.dice_loss(y_pred, y_true)
        return self.mse_weight * mse + self.bmse_weight * bone_mse + self.hmse_weight * hole_mse #+ self.smooth_weight*smooth_punish



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        :param preds: Model predictions, shape (N, ...)
        :param targets: Ground truth labels, shape (N, ...)
        :return: dice loss
        """
        # Apply sigmoid to ensure predictions are in the range [0, 1]
        preds = torch.sigmoid(preds)

        # Flatten the tensors
        preds = preds.view(-1)
        targets = targets.contiguous().view(-1)

        # Compute intersection
        intersection = (preds * targets).sum()

        # Calculate dice coefficient
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        # Return 1 - dice as the loss
        return 1 - dice


class mixMseLoss(nn.Module):
	def __init__(self):
		super(mixMseLoss, self).__init__()


	def forward(self, input, targets):
		input_np = input.cpu().detach().numpy() #(None, 1, 512)
		target_np = targets.cpu().detach().numpy()
		loss = 0
		for i in range(input_np.shape[0]):
			k_ignore = []
			for j in range(256):
				SE = 257**2
				k_tmp = 0
				for k in range(256):
					if k not in k_ignore:
						se_tmp = (target_np[i, 0, 2*j] - input_np[i, 0, 2*k])**2 + (target_np[i, 0, 2*j+1] - input_np[i, 0, 2*k+1])**2
						if se_tmp < SE:
							SE = se_tmp
							k_tmp = k
				# input_np[i, 0, 2*k_tmp] = input_np[i, 0, 2*k_tmp+1] = 512
				k_ignore.append(k_tmp)
				loss = loss + SE
		loss = np.array(loss/input_np.shape[0]/512)
		loss = torch.Tensor(loss)
		loss.requires_grad_(True)
		return torch.Tensor(loss)
    
