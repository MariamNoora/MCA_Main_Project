import os
import numpy as np
import nibabel as nib
import cv2
import torch
import torchio as tio
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Implements the 'Nogay & Adeli' preprocessing pipeline:
    1. Slice Extraction
    2. Canny Edge Detection (CED) & Auto-Cropping
    3. Restacking to 3D
    4. Normalization
    """
    def __init__(self, target_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.target_shape = target_shape

    def load_nifti(self, path: str) -> np.ndarray:
        """Loads .nii file and returns numpy array."""
        try:
            img = nib.load(path)
            data = img.get_fdata()
            
            # Handle RGB/structured data types from some NIfTI files (prevents VoidDType vs Float64DType errors)
            if data.dtype.names is not None:
                import numpy.lib.recfunctions as rfn
                data = rfn.structured_to_unstructured(data)
                data = np.mean(data, axis=-1)  # Convert RGB to grayscale
                
            # Ensure standard orientation if needed (skipped for speed, relies on MNI registration)
            return data
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise e

    def canny_crop_slice(self, slice_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Applies Canny Edge Detection to finding bounding box of the brain in a slice.
        Returns (x, y, w, h) or None if empty.
        """
        # Normalize slice to 0-255 for OpenCV
        if slice_img.max() == 0:
            return None
        
        norm_slice = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
        
        # Canny Edge Detection
        # Thresholds can be tuned. Grid search in paper suggests optimal values.
        # Standard starting point: 100, 200
        edges = cv2.Canny(norm_slice, 100, 200)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (assumed to be the brain)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        return x, y, w, h

    def process_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Applies CED to every slice and crops the 3D volume to the max bounding box found.
        Then resizes to target_shape.
        """
        # Volume shape: (H, W, D) usually. Let's assume (H, W, D).
        # We iterate through Depth.
        
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        has_brain = False
        
        # 1. Calculate Global Bounding Box across all slices
        for i in range(volume.shape[2]):
            slice_img = volume[:, :, i]
            bbox = self.canny_crop_slice(slice_img)
            if bbox:
                x, y, w, h = bbox
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
                has_brain = True
                
        if not has_brain:
            logger.warning("No brain detected in volume via Canny Edge. Returning resized original.")
            # Fallback: Just resize original
            cropped = volume
        else:
            # 2. Crop 3D Volume
            # We crop X and Y. We usually keep full Depth or also crop Depth?
            # Paper implies ROI extraction on slices.
            cropped = volume[y_min:y_max, x_min:x_max, :]
            
        # 3. Resize/Resample to Target Shape
        # Use Torchio for high-quality 3D resizing
        # tensor shape expectation: (Channels, H, W, D)
        tensor = torch.from_numpy(cropped).unsqueeze(0).float() # Add Channel dim
        
        transform = tio.Resize(self.target_shape)
        resized = transform(tensor)
        
        return resized.squeeze(0).numpy() # Return (H, W, D)


class Augmentor:
    """
    Implements 5x Augmentation Scheme:
    1. Original
    2. Flip (LR)
    3. Rotate 90
    4. Rotate 180
    5. Salt & Pepper Noise (5%)
    """
    def generate_versions(self, volume_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Input: (C, D, H, W) or (C, H, W, D).
        Returns list of 5 tensors.
        """
        versions = []
        
        # 1. Original
        versions.append(volume_tensor.clone())
        
        # 2. Flip (Left-Right) - Assuming axis 2 or 3 is width
        versions.append(torch.flip(volume_tensor, dims=[3])) # Try flipping last dim
        
        # 3. Rotate 90 (Spatial dims 2, 3)
        versions.append(torch.rot90(volume_tensor, k=1, dims=[2, 3]))
        
        # 4. Rotate 180
        versions.append(torch.rot90(volume_tensor, k=2, dims=[2, 3]))
        
        # 5. Salt Noise
        noise = torch.rand_like(volume_tensor)
        salt = (noise > 0.95).float() * volume_tensor.max()
        noisy = volume_tensor + salt
        # Clamp to max
        noisy = torch.clamp(noisy, 0, volume_tensor.max())
        versions.append(noisy)
        
        return versions

if __name__ == "__main__":
    # Test stub
    pass
