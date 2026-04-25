import os
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn import image
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_via_nilearn(data_dir="data/nilearn", n_subjects=50):
    """
    Uses Nilearn to fetch ABIDE data.
    Since Nilearn usually fetches 4D functional data, we will:
    1. Fetch func_preproc (4D)
    2. Compute Mean Image (3D) -> Save as .nii.gz
    3. Use this as our "Structural" proxy for the User Interface.
    """
    logger.info(f"Fetching {n_subjects} subjects via Nilearn...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch
    # pipeline='cpac', quality_checked=True
    abide = datasets.fetch_abide_pcp(
        data_dir=data_dir,
        n_subjects=n_subjects,
        pipeline='cpac',
        band_pass_filtering=True,
        global_signal_regression=True,
        derivatives=['func_preproc'],
        verbose=1
    )
    
    # Process to 3D
    logger.info("Processing 4D fMRI to 3D Mean Images...")
    
    # ABIDE dataset object has 'func_preproc' list and 'phenotypic' dataframe
    func_files = abide.func_preproc
    # Phenotypic is a record array
    pheno = abide.phenotypic
    
    # We need to link Sub ID to file to match our Dataset class expectation
    # Our Dataset expects: data_dir/{sub_id}.nii.gz
    
    export_dir = os.path.join(data_dir, "processed_3d")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save Phenotypic CSV
    # pheno is numpy recarray. Convert to DF.
    df = pd.DataFrame(pheno)
    csv_path = os.path.join(export_dir, "phenotypic.csv")
    df.to_csv(csv_path, index=False)
    
    count = 0
    for img_path, sub_id in zip(func_files, df['SUB_ID']):
        out_name = f"{sub_id}.nii.gz"
        out_path = os.path.join(export_dir, out_name)
        
        if os.path.exists(out_path):
            count += 1
            continue
            
        try:
            # Load 4D
            img = nib.load(img_path)
            # Compute Mean
            mean_img = image.mean_img(img)
            # Save 3D
            nib.save(mean_img, out_path)
            count += 1
            logger.info(f"Generated 3D mean for {sub_id}")
        except Exception as e:
            logger.error(f"Failed to process {sub_id}: {e}")
            
    logger.info(f"Done. {count} 3D images ready in {export_dir}")
    return export_dir, csv_path

if __name__ == "__main__":
    download_via_nilearn(n_subjects=20) # Start small to prove it works
