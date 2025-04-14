#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
"""

The following is the inference script for the baseline algorithm for Task 2 of the PANTHER challenge.

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
            mr_ip_dir_name = 'abdominal-t2-mri' #default value
        
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
            # Assuming you want to read the first .mha file found
            self.mr_image = mha_files[0]
        else:
            print('No mha images found in input directory')

    def run(self):
        """
        Load T2 MRI and generate segmentation of the tumor 
        """
        _show_torch_cuda_info()
        start_time = time.perf_counter()
        task = "Dataset091_PantherTask2"
        #move the checkpoints to the nnUNet_results directory
        self.move_checkpoints(self.weights_path, folds="0,1,2")

        #1. read the input image
        itk_image = sitk.ReadImage(self.mr_image)
        #   print shape and spacing
        print(f"Original image shape: {itk_image.GetSize()}, spacing: {itk_image.GetSpacing()}")
        #2. resample to low resolution and save as .nii.gz with _0000 ending for mrsegmentator
        low_res_image = resample_img(itk_image)
        sitk.WriteImage(low_res_image, str(self.mrsegmentator_input_dir / "mri_low_res_0000.nii.gz"))
        #3. run mrsegmentator 
        print("Input image for mrsegmentator:", os.listdir(self.mrsegmentator_input_dir))
        mrseg_image = os.path.join(self.mrsegmentator_input_dir, "mri_low_res_0000.nii.gz")
        inference.infer([mrseg_image], self.mrsegmentator_output_dir, [0])
        #   keep only the pancreas mask (pancreas==7)
        mrseg_mask = sitk.ReadImage(self.mrsegmentator_output_dir / "mri_low_res_0000_seg.nii.gz")
        mrseg_mask_array = sitk.GetArrayFromImage(mrseg_mask)
        pancreas_mask = mrseg_mask_array.copy()
        pancreas_mask[pancreas_mask != 7] = 0
        pancreas_mask[pancreas_mask == 7] = 1
        pancreas_mask_itk = sitk.GetImageFromArray(pancreas_mask)
        pancreas_mask_itk.CopyInformation(mrseg_mask)
        print(f"Pancreas mask unique values: {np.unique(pancreas_mask)}, mask shape: {pancreas_mask.shape}")
        #4. crop original image to pancreas region
        margins=[30, 30, 30]
        cropped_mri, crop_coordinates = CropPancreasROI(itk_image, pancreas_mask_itk, margins)
        #   save cropped image to nnunet input directory
        sitk.WriteImage(cropped_mri, self.nnunet_input_dir / "cropped_mri_0000.mha")
        #5. predict with nnUNet
        print(f"Input image nnunet:{os.listdir(self.nnunet_input_dir)}")
        mr_mask_name = "cropped_mri.mha"
        print(f"Input dir:{self.nnunet_input_dir}, output dir:{self.nnunet_output_dir}, task:{task}, folds :0,1,2")
        self.predict(
        input_dir=self.nnunet_input_dir,
        output_dir=self.nnunet_output_dir,
        task=task
        )
        print(f"Output files: {os.listdir(self.nnunet_output_dir)}")
        #6. Resample the nnUNet output to original resolution and save as final prediction
        tumor_mask_cropped = sitk.ReadImage(self.nnunet_output_dir / mr_mask_name)
        print(f"nnUNet output shape: {tumor_mask_cropped.GetSize()}, spacing: {tumor_mask_cropped.GetSpacing()}")
        final_tumor_mask = restore_to_full_size(tumor_mask_cropped, itk_image, crop_coordinates)
        print(f"Final tumor mask shape: {final_tumor_mask.GetSize()}, spacing: {final_tumor_mask.GetSpacing()}")
        sitk.WriteImage(final_tumor_mask, self.segmentation_mask)
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

