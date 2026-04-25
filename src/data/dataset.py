import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from .preprocessing import Preprocessor, Augmentor

class NeuroSpectrumDataset(Dataset):
    """
    Bio-Medical Dataset for ASD Classification.
    Features:
    - Loads raw NIfTI files
    - Matches with Phenotypic Data (Age, Gender, DX)
    - Applies Canny Edge Preprocessing
    - Implements 5x Augmentation Scheme (on-the-fly)
    """
    def __init__(self, root_dir: str, csv_path: str, transform=None):
        """
        Args:
            root_dir: Path to raw data (e.g., 'src/data/raw')
            csv_path: URL or Path to ABIDE Phenotypic CSV
        """
        self.root_dir = root_dir
        self.preprocessor = Preprocessor()
        self.augmentor = Augmentor()
        
        # Load Labels
        self.df = pd.read_csv(csv_path)
        
        # Scan filesystem for available subjects
        self.available_subjects = [] # List of (sub_id, path)
        
        # Walk through root_dir to find .nii.gz files
        # Expecting structure: root_dir/{sub_id}.nii.gz OR root_dir/{site}/{sub_id}..
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    # filename is usually {sub_id}.nii.gz or containing sub_id
                    # We assume basename is {sub_id}.nii.gz based on download script
                    sub_id_str = file.replace('.nii.gz', '')
                    try:
                        sub_id = int(sub_id_str)
                        self.available_subjects.append((sub_id, os.path.join(root, file)))
                    except ValueError:
                        continue
        
        print(f"Found {len(self.available_subjects)} scans in {root_dir}")

    def __len__(self):
        # 5x Expansion
        return len(self.available_subjects) * 5

    def __getitem__(self, idx):
        # Determine actual subject and augmentation version
        sub_idx = idx // 5
        aug_version = idx % 5
        
        sub_id, sub_path = self.available_subjects[sub_idx]
        
        # Get Labels
        row = self.df[self.df['SUB_ID'] == sub_id]
        if row.empty:
            # Fallback if label missing (should not happen in cleaned data)
            raise ValueError(f"No phenotypic data for Subject {sub_id}")
            
        dx_group = int(row['DX_GROUP'].values[0]) # 1=ASD, 2=Control
        sex = int(row['SEX'].values[0])           # 1=Male, 2=Female
        age = float(row['AGE_AT_SCAN'].values[0])
        
        # Map to Network Targets
        # Binary DX: ASD (1) -> 1, Control (2) -> 0? Or 0/1. 
        # Standard: ASD=1, Control=0
        label_dx = 1 if dx_group == 1 else 0
        
        # Gender: Male=0, Female=1
        label_sex = 0 if sex == 1 else 1
        
        # Age: Child (<18)=0, Adult (>=18)=1
        label_age = 0 if age < 18 else 1
        
        # Octal Class (Combined)
        # 0: TD-Male-Child
        # 1: TD-Male-Adult
        # 2: TD-Female-Child
        # 3: TD-Female-Adult
        # 4: ASD-Male-Child ...
        # Encoding: 4*DX + 2*Sex + Age
        label_octal = (4 * label_dx) + (2 * label_sex) + label_age
        
        # Load Volume
        # Check if pre-processed cache exists? For now, load raw and process.
        # WARNING: Canny Edge + Resizing is slow. In production, we should cache the processed .pt files.
        # But for "Implementing", we do it online.
        volume = self.preprocessor.load_nifti(sub_path)
        
        # Preprocess (Canny Crop + Resize)
        # Ideally, we crop ONCE, cache it. 
        # For this code, we just do it.
        processed_vol = self.preprocessor.process_volume(volume)
        
        # Augment
        # We need to apply the specific version transformation
        # Passing tensor to augmentor
        tensor_vol = torch.from_numpy(processed_vol).unsqueeze(0) # (1, H, W, D) (Actually preprocessor returns 1,H,W,D? No, it returns numpy HWD)
        # Preprocessor returns (C, H, W, D) if I changed it? 
        # Let's check preprocessing.py... it returns `resized.squeeze(0).numpy()` -> (H, W, D)
        # So manual unsqueeze needed for augmentor?
        # Augmentor expects tensor.
        
        # Let's fix dimension logic on the fly:
        tensor_in = torch.from_numpy(processed_vol).unsqueeze(0) # (1, H, W, D)
        
        # Generate specific version
        # Augmentor generates ALL 5. That's wasteful.
        # Better to have `augmentor.apply(tensor, version_idx)`
        # But I implemented `generate_versions`. 
        # I'll use `generate_versions` and pick one. It's okay for now.
        versions = self.augmentor.generate_versions(tensor_in)
        final_tensor = versions[aug_version]
        
        return final_tensor, {
            'dx': label_dx,
            'sex': label_sex,
            'age': label_age,
            'octal': label_octal,
            'sub_id': sub_id
        }
