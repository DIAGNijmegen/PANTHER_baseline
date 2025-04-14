# Copyright 2025 Diagnostic Image Analysis Group, Radboud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# functions resample_img and CropPancreasROI are copied from: https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py

import numpy as np
import SimpleITK as sitk
import time
import os

from scipy.ndimage import binary_dilation

def resample_img(itk_image, out_spacing  = [3.0, 3.0, 6.0], is_label=False, out_size = [], out_origin = [], out_direction= []):
    """
    Resamples an ITK image to a specified voxel spacing, optionally adjusting its size, origin, and direction.

    This function modifies the spatial resolution of a given medical image by changing its voxel spacing. 
    It can be used for both intensity images (e.g., CT, MRI) and segmentation masks, using appropriate interpolation methods.

    Parameters:
    -----------
    itk_image : sitk.Image
        The input image in SimpleITK format.
    
    out_spacing : list of float, optional (default: [2.0, 2.0, 2.0])
        The desired voxel spacing in (x, y, z) directions (in mm).
    
    is_label : bool, optional (default: False)
        Whether the input image is a label/segmentation mask.
        - `False`: Uses B-Spline interpolation for smooth intensity images.
        - `True`: Uses Nearest-Neighbor interpolation to preserve label values.
    
    out_size : list of int, optional (default: [])
        The desired output image size (in voxels). If not provided, it is automatically computed 
        to preserve the original physical image dimensions.
    
    out_origin : list of float, optional (default: [])
        The desired output image origin (in physical space). If not provided, the original image origin is used.
    
    out_direction : list of float, optional (default: [])
        The desired output image orientation. If not provided, the original image direction is used.

    Returns:
    --------
    itk_image : sitk.Image
        The resampled image with the specified voxel spacing, size, origin, and direction.

    Notes:
    ------
    - The function ensures that the physical space of the image is preserved when resampling.
    - If `out_size` is not specified, it is automatically computed based on the original and target spacing.
    - If resampling a segmentation mask (`is_label=True`), nearest-neighbor interpolation is used to avoid label mixing.

    Example:
    --------
    ```python
    # Resample an MRI image to 1mm isotropic resolution
    resampled_img = resample_img(mri_image, out_spacing=[1.0, 1.0, 1.0])

    # Resample a segmentation mask (preserving labels)
    resampled_mask = resample_img(segmentation_mask, out_spacing=[1.0, 1.0, 1.0], is_label=True)
    ```
    """
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    

    if not out_size:
        out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    if not out_direction:
        out_direction = itk_image.GetDirection()
    resample.SetOutputDirection(out_direction)
    if not out_origin:
        out_origin = itk_image.GetOrigin()
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image

def CropPancreasROI(image, low_res_segmentation, margins):
    """
    Crops the pancreas region from the original high-resolution image based on a low-resolution segmentation mask.
    
    This function first extracts the bounding box of the pancreas in the low-resolution segmentation, 
    then converts the coordinates to physical space and maps them back to the original image resolution. 
    The final crop includes user-defined margins.

    Parameters:
    -----------
    image : sitk.Image
        The original high-resolution medical image (e.g., CT or MRI).
    
    low_res_segmentation : sitk.Image
        The low-resolution binary segmentation mask where the pancreas is labeled (e.g., 0 = background, 1 = pancreas).
    
    margins : tuple of (float, float, float)
        The margin (in mm) to add around the pancreas region in the x, y, and z dimensions.

    Returns:
    --------
    cropped_image : sitk.Image
        The cropped region of the original high-resolution image containing the pancreas, with margins applied.
    
    crop_coordinates : dict
        Dictionary containing the cropping indices in the original image resolution:
        - 'x_start': Start index in x-dimension
        - 'x_finish': End index in x-dimension
        - 'y_start': Start index in y-dimension
        - 'y_finish': End index in y-dimension
        - 'z_start': Start index in z-dimension
        - 'z_finish': End index in z-dimension

    Notes:
    ------
    - The function ensures that the cropping indices remain within the bounds of the original image.
    - The low-resolution segmentation mask must be a **binary mask** (only 0 and 1 values).
    - The transformation ensures anatomical alignment between the low-resolution segmentation and the high-resolution image.
    
    Example:
    --------
    ```python
    cropped_img, crop_coords = CropPancreasROI(original_ct, low_res_seg, margins=(5.0, 5.0, 5.0))
    ```
    """
     
    pancreas_mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    assert(len(np.unique(pancreas_mask_np))==2)    
    
    pancreas_mask_nonzeros = np.nonzero(pancreas_mask_np)
    
    min_x = min(pancreas_mask_nonzeros[2])
    min_y = min(pancreas_mask_nonzeros[1])
    min_z = min(pancreas_mask_nonzeros[0])
    
    max_x = max(pancreas_mask_nonzeros[2])
    max_y = max(pancreas_mask_nonzeros[1])
    max_z = max(pancreas_mask_nonzeros[0])
    
    start_point_coordinates = (int(min_x), int(min_y), int(min_z))
    finish_point_coordinates = (int(max_x), int(max_y), int(max_z))          
    
    start_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(start_point_coordinates)
    finish_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(finish_point_coordinates)
    
    start_point = image.TransformPhysicalPointToIndex(start_point_physical)
    finish_point = image.TransformPhysicalPointToIndex(finish_point_physical)


    spacing = image.GetSpacing()
    size = image.GetSize()
        
    marginx = int(margins[0]/spacing[0])
    marginy = int(margins[1]/spacing[1])
    marginz = int(margins[2]/spacing[2])
    
    x_start = max(0, start_point[0] - marginx)
    x_finish = min(size[0], finish_point[0] + marginx)
    y_start = max(0, start_point[1] - marginy)
    y_finish = min(size[1], finish_point[1] + marginy)
    z_start = max(0, start_point[2] - marginz)
    z_finish = min(size[2], finish_point[2] + marginz)
    
    cropped_image = image[x_start:x_finish, y_start:y_finish, z_start:z_finish]

    crop_coordinates = {'x_start': x_start,
                        'x_finish': x_finish,
                        'y_start': y_start,
                        'y_finish': y_finish,
                        'z_start': z_start,
                        'z_finish': z_finish}
      
    return cropped_image, crop_coordinates

def restore_to_full_size(cropped_mask, original_image, crop_coordinates):
    """
    Restores a cropped mask to the original image size by placing it at the correct position.
    
    Parameters:
    -----------
    cropped_mask : sitk.Image
        The mask predicted on the cropped region
    original_image : sitk.Image
        The original full-sized image (used for size reference)
    crop_coordinates : dict
        Dictionary with keys 'x_start', 'x_finish', 'y_start', 'y_finish', 'z_start', 'z_finish'
        as returned by CropPancreasROI function
    
    Returns:
    --------
    sitk.Image
        Full-sized mask with the cropped mask placed at the correct position
    """
    # Get the original image size
    original_size = original_image.GetSize()
    
    # Create a blank mask of the original size
    full_mask = sitk.Image(original_size, cropped_mask.GetPixelID())
    full_mask.CopyInformation(original_image)  # Copy metadata from original image
    
    # Extract coordinates from dictionary
    x_start = crop_coordinates['x_start']
    x_finish = crop_coordinates['x_finish']
    y_start = crop_coordinates['y_start']
    y_finish = crop_coordinates['y_finish']
    z_start = crop_coordinates['z_start']
    z_finish = crop_coordinates['z_finish']
    
    # Convert cropped mask to numpy array
    cropped_array = sitk.GetArrayFromImage(cropped_mask)
    
    # Convert full mask to numpy array (for easier manipulation)
    full_array = sitk.GetArrayFromImage(full_mask)
    
    # Place the cropped mask in the correct position in the full array
    # Note: SimpleITK and numpy have different axis ordering (z,y,x vs x,y,z)
    full_array[z_start:z_finish, y_start:y_finish, x_start:x_finish] = cropped_array
    
    # Convert back to SimpleITK image
    result_mask = sitk.GetImageFromArray(full_array)
    result_mask.CopyInformation(original_image)  # Copy metadata
    
    return result_mask
