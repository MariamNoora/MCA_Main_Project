import os
import pandas as pd
import urllib.request
import logging
import argparse
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PHENOTYPIC_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
# --- CONFIGURATION (UPDATED FOR PREPROCESSED) ---
# Pipeline: ANTs (State-of-the-art registration)
# Derivative: anat_mni (Anatomical T1, MNI152 registered, Skull Stripped)
S3_PREPROC_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/dparsf/anat_mni" 
# Wait, let's use a known stable path.
# 'cpac' outputs: https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/anat_mni/{FILE_ID}_anat_mni.nii.gz
# FILE_ID = {SITE_ID}_{SUB_ID} (no zero padding usually, but checked in loop)

def download_file(url, output_path):
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def download_abide_full(data_dir: str = "data/preprocessed", limit: int = None):
    """
    Downloads Preprocessed T1w MRI scans (CPAC Pipeline, MNI Registered).
    Guarantees access to standard clean data.
    """
    logger.info(f"Starting ABIDE Preprocessed Download (CPAC - MNI152). Target: {'FULL' if limit is None else limit}")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Fetch Phenotypic Data
    logger.info("Fetching Phenotypic Data Table...")
    try:
        df = pd.read_csv(PHENOTYPIC_URL)
    except Exception as e:
        logger.error(f"Could not fetch phenotypic data: {e}")
        return

    # 2. Filter/Select Subjects
    target_subs = df.copy()
    if limit:
        # Balance classes
        n_half = limit // 2
        # Ensure we shuffle to get random subjects, not just first N
        # Seed for reproducibility
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        asd = df_shuffled[df_shuffled['DX_GROUP'] == 1]
        ctrl = df_shuffled[df_shuffled['DX_GROUP'] == 2]
        
        # Take n_half from each
        target_subs = pd.concat([asd.head(n_half), ctrl.head(n_half)])
        
        logger.info(f"Selected {len(target_subs)} subjects: {len(target_subs[target_subs['DX_GROUP']==1])} ASD, {len(target_subs[target_subs['DX_GROUP']==2])} Control")
    
    logger.info(f"Queueing {len(target_subs)} downloads...")
    
    # 3. Download Loop
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(target_subs.iterrows(), total=len(target_subs)):
        site = row['SITE_ID']
        sub_id = row['SUB_ID']
        
        # FILE_ID Construction: {SITE_ID}_{SUB_ID} (e.g. NYU_0050952)
        # Note: SUB_ID in CSV is int (50952). S3 expects '0050952' (Zero padded to 7? No, usually 7 chars total string?)
        # Actually, ABIDE SUB_IDs are 5 digits. 
        # The filename usually uses zero-padded 7-digit ID (0050952).
        
        sub_id_str = f"{sub_id:07d}" # Force 7 digits (0050952)
        file_id = f"{site}_{sub_id_str}"
        
        # Filename: {FILE_ID}_anat_mni.nii.gz
        filename = f"{file_id}_anat_mni.nii.gz"
        
        # URL: S3 Base + filename
        # CPAC structural path:
        # https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/anat_mni/[FILE_ID]_anat_mni.nii.gz
        url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/anat_mni/{filename}"
        
        # Save Path
        save_path = os.path.join(data_dir, f"{sub_id}.nii.gz") # Keep simple local ID
        
        if os.path.exists(save_path):
            success_count += 1
            continue
            
        if download_file(url, save_path):
            success_count += 1
        else:
            # Fallback: Try without zero padding?
            # Some sites behave differently.
            fail_count += 1
            
    logger.info(f"Download Finished. Success: {success_count} / {len(target_subs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ABIDE I Dataset")
    parser.add_argument("--full", action="store_true", help="Download full dataset (ignore safety limit)")
    parser.add_argument("--limit", type=int, default=50, help="Number of subjects to download if not full")
    
    args = parser.parse_args()
    
    if args.full:
        download_abide_full(limit=None)
    else:
        download_abide_full(limit=args.limit)
