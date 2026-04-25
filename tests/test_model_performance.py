import sys
import os
import torch
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import time
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model.system import NeuroSpectrumSystem
from data.preprocessing import Preprocessor

def get_best_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        return None
        
    # Heuristic: Find 'best' by val_acc_octal
    def get_acc_from_name(name):
        try:
            return float(name.split("val_acc_octal=")[1].split("-")[0].replace(".ckpt", ""))
        except:
            return -1.0
            
    best_ckpt = max(checkpoints, key=get_acc_from_name)
    return os.path.join(checkpoint_dir, best_ckpt)

def get_ground_truth_label(row):
    """
    Maps phenotypic row to Octal Class Index (0-7).
    
    Classes:
    0: TD-Male-Child
    1: TD-Male-Adult
    2: TD-Female-Child
    3: TD-Female-Adult
    4: ASD-Male-Child
    5: ASD-Male-Adult
    6: ASD-Female-Child
    7: ASD-Female-Adult
    
    Data Mapping:
    DX_GROUP: 1=ASD, 2=TD
    SEX: 1=Male, 2=Female
    AGE_AT_SCAN: <18 Child, >=18 Adult
    """
    try:
        dx = int(row['DX_GROUP']) # 1=ASD, 2=TD
        sex = int(row['SEX']) # 1=Male, 2=Female
        age = float(row['AGE_AT_SCAN'])
        
        is_asd = (dx == 1)
        is_male = (sex == 1)
        is_child = (age < 18.0)
        
        # Mapping Logic:
        # Base: 0
        # ASD: +4 (Indices 4-7)
        # Female: +2 (Indices 2,3,6,7)
        # Adult: +1 (Indices 1,3,5,7)
        
        idx = 0
        if is_asd: idx += 4
        if not is_male: idx += 2
        if not is_child: idx += 1
        
        return idx
    except Exception as e:
        # print(f"Error parse GT: {e}")
        return -1

def get_class_name(idx):
    labels = [
        "TD-Male-Child", "TD-Male-Adult", "TD-Female-Child", "TD-Female-Adult",
        "ASD-Male-Child", "ASD-Male-Adult", "ASD-Female-Child", "ASD-Female-Adult"
    ]
    if 0 <= idx < len(labels):
        return labels[idx]
    return "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 1. Setup
    # Path relative to this script: ../data/nilearn/processed_3d
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data/nilearn/processed_3d")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    csv_path = os.path.join(data_dir, "phenotypic.csv")
    
    ckpt_path = get_best_checkpoint(checkpoint_dir)
    
    if not os.path.exists(csv_path):
        print(f"Phenotypic CSV not found at {csv_path}")
        return
    if not ckpt_path:
        print(f"No checkpoint found in {checkpoint_dir}")
        return

    print(f"Loading Model from: {ckpt_path}")
    device = torch.device(args.device)
    print(f"Device: {device}")
    
    # Load Model
    try:
        model = NeuroSpectrumSystem.load_from_checkpoint(ckpt_path)
        model.to(device)
        model.eval()
        model.freeze()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} subject records.")
    
    # Check Preprocessor
    preprocessor = Preprocessor()
    
    # Scan files
    files = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    print(f"Found {len(files)} NIfTI files.")
    
    correct_count = 0
    processed_count = 0
    
    print("\nStarting Inference...")
    print("-" * 120)
    print(f"{'Sub ID':<10} | {'True Class':<20} | {'Pred Class':<20} | {'Conf':<6} | {'Result'} | {'Correctness'}")
    print("-" * 120)

    for file_path in files:
        sub_id_str = os.path.basename(file_path).split('.')[0]
        try:
            sub_id = int(sub_id_str)
        except:
            continue
            
        # Get GT
        row = df[df['SUB_ID'] == sub_id]
        if row.empty:
            continue
        row = row.iloc[0]
        
        gt_idx = get_ground_truth_label(row)
        if gt_idx == -1:
            continue
            
        # Preprocess
        try:
            # Load
            raw_vol = preprocessor.load_nifti(file_path)
            # Process (Canny + Resize)
            processed_vol = preprocessor.process_volume(raw_vol) # (128,128,128)
            
            # Prepare Tensor: (1, 1, 128, 128, 128)
            input_tensor = torch.from_numpy(processed_vol).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs['octal_logits']
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            is_correct = (pred_idx == gt_idx)
            if is_correct: correct_count += 1
            processed_count += 1
            
            res_str = "✅ PASS" if is_correct else "❌ FAIL"
            
            gt_name = get_class_name(gt_idx)
            pred_name = get_class_name(pred_idx)
            
            print(f"{sub_id:<10} | {gt_name:<20} | {pred_name:<20} | {confidence:.2f}   | {res_str:<6} | {is_correct}")
            
        except Exception as e:
            print(f"Error processing {sub_id}: {e}")

    print("-" * 120)
    if processed_count > 0:
        acc = correct_count / processed_count * 100
        print(f"\nFinal Accuracy: {acc:.2f}% ({correct_count}/{processed_count})")
    else:
        print("\nNo subjects processed successfully.")

if __name__ == "__main__":
    main()
