import os
import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import argparse

from data.dataset import NeuroSpectrumDataset
from model.system import NeuroSpectrumSystem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    L.seed_everything(42)
    
    # 0. Configuration
    # PHENOTYPIC_URL can be local now
    PHENOTYPIC_PATH = os.path.join(os.path.dirname(__file__), "../data/nilearn/processed_3d/phenotypic.csv")
    
    # Path Handling
    # Using 'data/nilearn/processed_3d'
    root_dir = os.path.join(os.path.dirname(__file__), "../data/nilearn/processed_3d")
    
    # 1. Setup Data
    print(f"Loading Dataset from {root_dir}...")
    try:
        # Check if CSV exists, if not maybe nilearn is still running.
        # But Dataset class needs a path.
        if not os.path.exists(PHENOTYPIC_PATH):
             # Fallback to URL if local not ready? 
             # No, if local data is nilearn, we need nilearn csv.
             print("Waiting for Nilearn download...")
        
        dataset = NeuroSpectrumDataset(root_dir=root_dir, csv_path=PHENOTYPIC_PATH)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print("Ensure 'data/raw' exists and contains .nii.gz files.")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Run 'python data/download_abide.py' first.")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Loaders
    # Num workers=0 for Windows compatibility
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=0)
    
    # 2. Setup System
    system = NeuroSpectrumSystem(lr=1e-4)
    
    # 3. Trainer
    logger = TensorBoardLogger("logs", name="neuro_spectrum")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_octal',
        dirpath='checkpoints',
        filename='neuro-spectrum-{epoch:02d}-{val_acc_octal:.2f}',
        save_top_k=3,
        mode='max',
    )
    
    trainer = L.Trainer(
        max_epochs=50,
        accelerator='auto', # GPU if available
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        log_every_n_steps=10
    )
    
    # 4. Train
    print("Starting Training...")
    trainer.fit(system, train_loader, val_loader, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    main()
