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

# function resample_img is copied from: https://github.com/DIAGNijmegen/PANORAMA_baseline/blob/main/src/data_utils.py

def calculate_overlap(pancreas_mask, component_mask):
	"""
	Calculate the overlap between a pancreas mask and a component mask.
	
	Args:
		pancreas_mask (numpy.ndarray): Binary mask of the pancreas
		component_mask (numpy.ndarray): Binary mask of the connected component
		
	Returns:
		tuple: (overlap_voxels, overlap_percentage)
			- overlap_voxels: Number of overlapping voxels
			- overlap_percentage: Percentage of pancreas covered by the component
	"""
	import numpy as np
	
	# Ensure both are binary masks
	pancreas_mask = pancreas_mask > 0
	component_mask = component_mask > 0
	
	# Calculate overlap (intersection)
	overlap_voxels = np.sum(pancreas_mask & component_mask)
	
	# Calculate percentage of pancreas covered by the component
	pancreas_voxels = np.sum(pancreas_mask)
	component_voxels = np.sum(component_mask)
	if component_voxels == 0:
		overlap_percentage = 0.0
	else:
		overlap_percentage = 100.0 * overlap_voxels / component_voxels
	
	return overlap_voxels, overlap_percentage

def filter_components(labels, num_features, pancreas_mask, min_overlap_percent=15.0, min_voxels=10):
	"""
	Filter connected components based on minimum overlap with pancreas mask and minimum size.
	
	Args:
		labels (numpy.ndarray): Array with labeled connected components
		num_features (int): Number of connected components 
		pancreas_mask (numpy.ndarray): Binary mask of the pancreas
		min_overlap_percent (float): Minimum overlap percentage threshold
		min_voxels (int): Minimum number of voxels a component must have
		
	Returns:
		numpy.ndarray: Binary mask with only components meeting the threshold
	"""
	import numpy as np
	
	# Create an empty mask for the filtered result
	filtered_mask = np.zeros_like(labels, dtype=bool)
	
	# Track which components meet the criteria
	kept_components = []
	
	for i in range(1, num_features+1):
		component = (labels == i)
		component_voxels = np.sum(component)
		overlap_voxels, overlap_percentage = calculate_overlap(pancreas_mask, component)
		
		# Check both conditions: minimum overlap percentage and minimum size
		if overlap_percentage >= min_overlap_percent and component_voxels >= min_voxels:
			# Add this component to our filtered mask
			filtered_mask = filtered_mask | component
			kept_components.append(i)
	
	print(f"Kept {len(kept_components)}/{num_features} components: {kept_components}")
	return filtered_mask

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
	import SimpleITK as sitk
	import numpy as np
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

def upsample_mask(low_res_mask, source_spacing, target_spacing, fill_holes=True):
	"""
	Upsample a low-resolution mask using spacing information.
	
	Parameters:
	-----------
	low_res_mask : numpy.ndarray
		The segmentation mask predicted at low resolution.
	source_spacing : tuple or list of float
		Spacing of the low-resolution mask (e.g., (z, y, x)).
	target_spacing : tuple or list of float
		Spacing of the target image (e.g., (z, y, x)).
	fill_holes : bool, optional
		Whether to fill holes in the upsampled mask.
		
	Returns:
	--------
	upsampled_mask : numpy.ndarray
		The mask upsampled to the target spacing.
	"""
	import numpy as np
	from scipy import ndimage

	# Compute zoom factor for each axis: ratio of source_spacing to target_spacing.
	zoom_factors = [s / t for s, t in zip(source_spacing, target_spacing)]
	
	# Upsample using nearest neighbor interpolation (order=0).
	upsampled_mask = ndimage.zoom(low_res_mask, zoom=zoom_factors, order=0)
	
	if fill_holes:
		# If binary segmentation, fill holes slice by slice.
		# For multi-class segmentation, you'd need to iterate over each label.
		upsampled_mask = ndimage.binary_fill_holes(upsampled_mask).astype(upsampled_mask.dtype)
	
	return upsampled_mask