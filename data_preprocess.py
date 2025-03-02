import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import shutil


def get_file_names(path, filetype):
    """
    Recursively find all files with the specified file type in a directory and its subdirectories.

    :param path: Root directory to start the search.
    :param filetype: The file extension (e.g., '.csv') to look for.
    :return: A list of full file paths for files with the specified file extension.
    """
    names = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                # Use os.path.join to handle path concatenation across different OSes
                names.append(os.path.join(root, file))
    return names


def copy_and_rename(src_file_path, dest_folder_path, new_file_name):
    """
    Copy a file to the specified folder and rename it.
    
    :param src_file_path: Source file path
    :param dest_folder_path: Destination folder path
    :param new_file_name: New file name (including extension)
    """
    try:
        # Ensure the destination folder exists
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)
        
        # Construct the full path for the destination file
        dest_file_path = os.path.join(dest_folder_path, new_file_name)
        
        # Use shutil.copy() to copy the file to the destination folder and rename it
        shutil.copy(src_file_path, dest_file_path)
        
        # print(f"File successfully copied and renamed to: {dest_file_path}")
    except IOError as e:
        print(f"Error occurred while copying the file: {e.strerror}")
    except Exception as e:
        print(f"An unknown error occurred: {e}")


def pad_image_to_size(image_np, target_size=(216, 96, 96)):
    """
    Pads a 3D NumPy array to the target size with zeros.

    Parameters:
    image_np (ndarray): A 3D NumPy array representing the image.
    target_size (tuple): The target size (x, y, z) to pad to.

    Returns:
    ndarray: The padded 3D NumPy array with the specified target size.
    """
    # Get the current size of the image
    current_size = image_np.shape
    
    # Create a new array filled with zeros of the target size
    padded_image_np = np.zeros(target_size, dtype=image_np.dtype)
    
    # Calculate the valid range for placing the original image
    start = [(target_size[i] - current_size[i]) // 2 for i in range(3)]
    end = [start[i] + current_size[i] for i in range(3)]
    
    # Place the original image in the center of the padded array
    padded_image_np[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = image_np
    
    return padded_image_np



def crop_around_island(image, crop_size):
    """
    Crop an image around its non-zero "island" center with a specified crop size.

    Parameters:
    image (SimpleITK.Image): The input image.
    crop_size (tuple): Desired output size as (x_size, y_size, z_size).
    
    Returns:
    SimpleITK.Image: The cropped image.
    """
    image_np = sitk.GetArrayFromImage(image)  # Convert to NumPy array for easier manipulation

    # Find the non-zero voxel coordinates (the "island")
    non_zero_coords = np.argwhere(image_np != 0)

    if len(non_zero_coords) == 0:
        print("Warning: No non-zero voxels found in the image.")
        return None  # 或者返回原始图像或其他默认值

    # Calculate the center of the island
    center_of_mass = np.mean(non_zero_coords, axis=0).astype(int)

    # Crop size should be half on each side from the center
    crop_half_size = [size // 2 for size in crop_size]

    # Define the start and end indices for the crop
    start_indices = [max(0, center_of_mass[i] - crop_half_size[i]) for i in range(3)]
    end_indices = [min(image_np.shape[i], center_of_mass[i] + crop_half_size[i]) for i in range(3)]

    # Crop the image using the computed start and end indices
    cropped_np = image_np[
        start_indices[0]:end_indices[0],
        start_indices[1]:end_indices[1],
        start_indices[2]:end_indices[2]
    ]

    # Convert back to SimpleITK image
    cropped_image = sitk.GetImageFromArray(cropped_np)

    # Manually copy the metadata (spacing, origin, direction)
    cropped_image.SetSpacing(image.GetSpacing())
    cropped_image.SetOrigin(image.GetOrigin())
    cropped_image.SetDirection(image.GetDirection())
    return cropped_image


def register_images(fixed_image, moving_image):
    # Registers a moving image to a fixed image using SimpleITK.
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=50, convergenceMinimumValue=1e-1, convergenceWindowSize=5)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    
    return resampled_image



if __name__ == '__main__':
    crop_size = (216, 96, 96)
    file_names = get_file_names('./data0930','.gz')
    idx = 0
    for _, file_name in tqdm(enumerate(file_names, 0), total=len(file_names) - 1, desc="\033[31mdata preprocessing:\033[0m"):
        if 'PRE' in file_name:
            pre_name = file_name
            post_name = file_name.replace('PRE', 'POST')
            if os.path.isfile(pre_name) and os.path.isfile(post_name):
                pre_crop = crop_around_island(sitk.ReadImage(pre_name, sitk.sitkFloat32), crop_size)
                post_crop = crop_around_island(sitk.ReadImage(post_name, sitk.sitkFloat32), crop_size)
                post_regis = register_images(pre_crop, post_crop)
                if idx % 5 == 0:
                    sitk.WriteImage(pre_crop, f'./dataset/val/img/{idx}.nii.gz')
                    sitk.WriteImage(post_regis, f'./dataset/val/label/{idx}.nii.gz')
                else:
                    sitk.WriteImage(pre_crop, f'./dataset/train/img/{idx}.nii.gz')
                    sitk.WriteImage(post_regis, f'./dataset/train/label/{idx}.nii.gz')
                idx+=1
                
                
