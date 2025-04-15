#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
The following is the inference script for the baseline algorithm for Task 1 of the PANTHER challenge.

It is meant to run within a container.

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the GC platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

from pathlib import Path
import time
from mrsegmentator import inference
import glob
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import shutil
from scipy import ndimage
from data_utils import *

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import warnings
warnings.filterwarnings("ignore")
class PancreaticTumorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.nnunet_input_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/nnUNet_results")
        self.mrsegmentator_input_dir = Path("/opt/algorithm/mrsegmentator/input")
        self.mrsegmentator_output_dir = Path("/opt/algorithm/mrsegmentator/output")

        # input / output paths for predictions-model
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print("Folder containing eval image", mr_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_mri))
            mr_ip_dir_name = 'abdominal-t1-mri' #default value
        
        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}") #abdominal-t2-mri
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "pancreatic-tumor-segmentation"))
        self.segmentation_mask = self.output_dir_seg_mask / "tumor_seg.mha"
        self.weights_path = Path("/opt/ml/model") #weights can be uploaded as a separate tarball to Grand Challenge (Algorithm > Models). The resources will be extracted to this path at runtime
        self.mrsegmentator_weights = "/opt/ml/model/weights"
        os.environ["MRSEG_WEIGHTS_PATH"] = self.mrsegmentator_weights

        # ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_input_dir.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_output_dir.mkdir(exist_ok=True, parents=True)

        mha_files = glob.glob(os.path.join(self.mr_ip_dir, '*.mha'))
        # Check if any .mha files were found
        if mha_files:
            self.mr_image = mha_files[0]
        else:
            print('No mha images found in input directory')

    def run(self):
        """
        Load T1 MRI and generate segmentation of the tumor 
        """
        _show_torch_cuda_info()
        start_time = time.perf_counter()
        task = "Dataset090_PantherTask1"
        #move the checkpoints to the nnUNet_results directory
        self.move_checkpoints(self.weights_path, folds="0,1,2",task=task)

        #1. copy the input image to nnUNet input directory adding _0000 suffix
        shutil.copyfile(self.mr_image, self.nnunet_input_dir / "mri_0000.mha")
        #   print shape and spacing
        itk_image = sitk.ReadImage(self.mr_image)
        print(f"Original image shape: {itk_image.GetSize()}, spacing: {itk_image.GetSpacing()}")
        #2. Predict with nnUNet
        print(f"Input image nnunet:{os.listdir(self.nnunet_input_dir)}")
        mr_mask_name = "mri.mha"
        print(f"Input dir:{self.nnunet_input_dir}, output dir:{self.nnunet_output_dir}, task:{task}, folds :0,1,2")
        self.predict(
        input_dir=self.nnunet_input_dir,
        output_dir=self.nnunet_output_dir,
        task=task
        )
        print(f"Output files: {os.listdir(self.nnunet_output_dir)}")
        #3. Post-process nnUNet output
        #   read the nnUNet output
        tumor_mask = sitk.ReadImage(self.nnunet_output_dir / mr_mask_name)
        print(f"nnUNet output shape: {tumor_mask.GetSize()}, spacing: {tumor_mask.GetSpacing()}")
        #   convert to numpy array
        tumor_mask_array = sitk.GetArrayFromImage(tumor_mask)
        #   find connected components
        labels, num_features = ndimage.label(tumor_mask_array)
        print(f"Number of connected components: {num_features}")
        #   if more than 0 connected component are predicted post-process is applied
        if num_features > 0:
            print(f"Post-processing nnUNet output: {num_features} connected components")
            #    resample the image to half the original spacing for faster inference with mrsegmentator
            print(f"Resampling image to half the original spacing")
            out_spacing = [itk_image.GetSpacing()[0] * 2, itk_image.GetSpacing()[1] * 2, itk_image.GetSpacing()[2] * 2]
            print(f"Original spacing: {itk_image.GetSpacing()}, resampled spacing: {out_spacing}")
            image_low_res = resample_img(itk_image, out_spacing  = out_spacing, is_label=False)
            sitk.WriteImage(image_low_res, str(self.mrsegmentator_input_dir / "mri_0000.nii.gz"))
            mrseg_image = os.path.join(self.mrsegmentator_input_dir, "mri_0000.nii.gz")
            #   generate pancreas mask with mrsegmentator
            inference.infer([mrseg_image], self.mrsegmentator_output_dir, [0])
            #   keep only the pancreas mask (pancreas==7)
            mrseg_mask = sitk.ReadImage(self.mrsegmentator_output_dir / "mri_0000_seg.nii.gz")
            mrseg_mask_array = sitk.GetArrayFromImage(mrseg_mask)
            pancreas_mask = mrseg_mask_array.copy()
            pancreas_mask[pancreas_mask != 7] = 0
            pancreas_mask[pancreas_mask == 7] = 1
            pancreas_mask_high_res = upsample_mask(pancreas_mask, mrseg_mask.GetSpacing(), itk_image.GetSpacing(), fill_holes=False)
            #   keep only the connected components that overlap with the pancreas mask at least 15% and have at least 10 voxels
            print("Filtering connected components based on overlap with pancreas mask")
            filtered_tumor_mask = filter_components(labels, num_features, pancreas_mask_high_res)
            #   save the filtered mask
            filtered_tumor_mask = filtered_tumor_mask.astype(np.uint8)
            print(f"Filtered tumor mask unique values: {np.unique(filtered_tumor_mask)}")
            filtered_tumor_mask_itk = sitk.GetImageFromArray(filtered_tumor_mask)
            filtered_tumor_mask_itk.CopyInformation(tumor_mask)
            print(f"Filtered tumor mask shape: {filtered_tumor_mask_itk.GetSize()}, spacing: {filtered_tumor_mask_itk.GetSpacing()}")
            sitk.WriteImage(filtered_tumor_mask_itk, self.segmentation_mask)

        else:
            print(f"nnUNet output has no connected components, no post-processing applied")
            shutil.copyfile(self.nnunet_output_dir / mr_mask_name, self.segmentation_mask)

        end_time = time.perf_counter()
        print(f"Prediction time: {end_time - start_time:.3f} seconds")
    
    def predict(self, input_dir, output_dir, task="Dataset091_PantherTask2", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2"):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
                fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
                cmd.extend(fold_list)

            if checkpoint:
                cmd.append('-chk')
                cmd.append(str(checkpoint))

            cmd_str = " ".join(cmd)
            print(f"Running command: {cmd_str}")
            subprocess.check_call(cmd_str, shell=True)

    def move_checkpoints(self, source_dir, folds="0,1,2", trainer="nnUNetTrainer", task="Dataset091_PantherTask2"):
        """
        Move nnUNet checkpoints to nnUNet_results directory.
        """
        # Create the top-level destination directory if it doesn't exist
        os.makedirs(self.nnunet_model_dir, exist_ok=True)
        print(os.listdir(source_dir))
        task_name = task.split("_")[1]
        
        # Determine fold list, supporting both a comma-separated string or a single value.
        fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [str(folds)]
        
        # Move the checkpoints
        for fold in fold_list:
            source_path = os.path.join(source_dir, f"checkpoint_best_{task_name}_fold_{fold}.pth")
            destination_path = os.path.join(self.nnunet_model_dir, task, f"{trainer}__nnUNetPlans__3d_fullres", f"fold_{fold}", "checkpoint_final.pth")
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Move the file
            try:
                shutil.copyfile(source_path, destination_path)
                print(f"Copied checkpoint for fold {fold} to {destination_path}")
            except FileNotFoundError:
                print(f"Source file not found: {source_path}")
            except Exception as e:
                print(f"Error moving checkpoint for fold {fold}: {e}")


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()
