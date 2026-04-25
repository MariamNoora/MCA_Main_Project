
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.model.system import NeuroSpectrumSystem
from src.data.dataset import NeuroSpectrumDataset

def evaluate_paper_metrics(ckpt_path, root_dir, csv_path):
    """
    Evaluates the model on:
    1. Gender-based Quadruple Classification (F-ASD, F-TD, M-ASD, M-TD)
    2. Age-based Quadruple Classification (Child-ASD, Child-TD, Adult-ASD, Adult-TD)
    3. Octal Classification (All 8 combinations)
    """
    
    # Load Model
    print(f"Loading checkpoint: {ckpt_path}")
    system = NeuroSpectrumSystem.load_from_checkpoint(ckpt_path)
    system.eval()
    system.freeze()
    
    # Load Data (No Augmentations for Evaluation to keep distinct subjects)
    # Actually, we need to be careful. The dataset class augments by default.
    # We should use the first version (augmented version 0) or average predictions?
    # For simplicity, let's use the standard dataset but only take the first version (index % 5 == 0)
    
    dataset = NeuroSpectrumDataset(root_dir=root_dir, csv_path=csv_path)
    
    # Filter for validation (or just use all data if we re-downloaded everything for this purpose)
    # The paper uses 5-fold cross validation. We are doing a hold-out test here.
    # Let's just evaluate on the whole dataset to see "training/validation" performance for now,
    # as we don't have a separate test set split defined in file structure.
    
    print(f"Evaluated on {len(dataset)} samples (including augmentations).")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_preds_gender = []
    all_preds_age = []
    all_preds_octal = []
    
    all_labels_gender = []
    all_labels_age = []
    all_labels_octal = []
    all_labels_dx = []
    all_labels_sex = []
    all_labels_age_group = []

    print("Running Inference...")
    for batch in tqdm(dataloader):
        x, labels = batch
        outputs = system(x)
        
        # Predictions
        pred_gender = torch.softmax(outputs['gender_logits'], dim=1).argmax(1).cpu().numpy()
        pred_age = torch.softmax(outputs['age_logits'], dim=1).argmax(1).cpu().numpy()
        pred_octal = torch.softmax(outputs['octal_logits'], dim=1).argmax(1).cpu().numpy()
        
        all_preds_gender.extend(pred_gender)
        all_preds_age.extend(pred_age)
        all_preds_octal.extend(pred_octal)
        
        # Ground Truth
        all_labels_dx.extend(labels['dx'].cpu().numpy())
        all_labels_sex.extend(labels['sex'].cpu().numpy()) # 0=M, 1=F (Need to verify dataset.py mapping!)
        all_labels_age_group.extend(labels['age'].cpu().numpy()) # 0=Child, 1=Adult
        all_labels_octal.extend(labels['octal'].cpu().numpy())

    # --- Verify Mappings from src/data/dataset.py ---
    # label_dx = 1 if dx_group == 1 (ASD) else 0 (TD)
    # label_sex = 0 if sex == 1 (Male) else 1 (Female)
    # label_age = 0 if age < 18 (Child) else 1 (Adult)
    
    # --- Derived Labels ---
    # We need to Construct the "Quadruple Labels" from the binary components
    # Model 1 (Gender w/ DX): 
    # Classes: 0: M-TD, 1: M-ASD, 2: F-TD, 3: F-ASD (Example mapping)
    # Actually, let's look at how the model was trained. 
    # The 'gender_head' in resnet3d_multiclass.py outputs 2 classes? or 4?
    # Let's check the code. IMPORTANT.
    # If the model only outputs Gender (M/F), we cannot do Quadruple Classification directly from that head.
    # **Hypothesis**: The user wants us to use the **Octal Head** to derive these or check if we trained 3 separate 4-class heads.
    # The current `NeuroSpectrumSystem` has:
    # self.gender_head = nn.Linear(512, 2)
    # self.age_head = nn.Linear(512, 2)
    # self.octal_head = nn.Linear(512, 8)
    
    # **CORRECTION**: To match the paper, we should have trained:
    # Head 1: 4 classes (Sex x DX)
    # Head 2: 4 classes (Age x DX)
    # Head 3: 8 classes (Sex x Age x DX)
    
    # Since we implemented `gender_head` as just Gender (2 classes), we can't directly evaluate "Gender-based ASD diagnosis" from it
    # UNLESS we use the **Octal Head** and collapse the classes!
    
    # Strategy: Use Octal Predictions to derive Gender-Quad and Age-Quad predictions.
    # Octal Mapping (from dataset.py):
    # label_octal = (4 * label_dx) + (2 * label_sex) + label_age
    # 0: TD, Male, Child
    # 1: TD, Male, Adult
    # 2: TD, Female, Child
    # 3: TD, Female, Adult
    # 4: ASD, Male, Child
    # 5: ASD, Male, Adult
    # 6: ASD, Female, Child
    # 7: ASD, Female, Adult
    
    # Model 1 (Gender-based Quadruple): Group by Gender & DX (Ignore Age)
    # Classes: M-TD, M-ASD, F-TD, F-ASD
    # M-TD: {0, 1}
    # M-ASD: {4, 5}
    # F-TD: {2, 3}
    # F-ASD: {6, 7}
    
    # Model 2 (Age-based Quadruple): Group by Age & DX (Ignore Gender)
    # Classes: Child-TD, Child-ASD, Adult-TD, Adult-ASD
    # Child-TD: {0, 2}
    # Child-ASD: {4, 6}
    # Adult-TD: {1, 3}
    # Adult-ASD: {5, 7}
    
    # We will compute accuracies based on these groupings from the *Octal Predictions*.
    
    print("\n--- Results Analysis ---")
    
    # Convert lists to numpy
    preds = np.array(all_preds_octal)
    truth = np.array(all_labels_octal)
    
    # 1. Octal Accuracy (Model 3)
    acc_octal = accuracy_score(truth, preds)
    print(f"Model 3 (Octal Classification) Accuracy: {acc_octal*100:.2f}% (Paper: 67.94%)")
    
    # 2. Gender-Quad Accuracy (Model 1)
    # Map 8 classes to 4 classes (M-TD, M-ASD, F-TD, F-ASD)
    def map_to_gender_quad(k):
        # 0,1 -> 0 (M-TD)
        # 4,5 -> 1 (M-ASD)
        # 2,3 -> 2 (F-TD)
        # 6,7 -> 3 (F-ASD)
        if k in [0, 1]: return 0
        if k in [4, 5]: return 1
        if k in [2, 3]: return 2
        if k in [6, 7]: return 3
        return -1

    preds_gq = np.array([map_to_gender_quad(p) for p in preds])
    truth_gq = np.array([map_to_gender_quad(t) for t in truth])
    
    acc_gq = accuracy_score(truth_gq, preds_gq)
    print(f"Model 1 (Gender Quadruple) Accuracy:   {acc_gq*100:.2f}% (Paper: 80.94%)")
    
    # 3. Age-Quad Accuracy (Model 2)
    # Map 8 classes to 4 classes (Child-TD, Child-ASD, Adult-TD, Adult-ASD)
    def map_to_age_quad(k):
        # 0,2 -> 0 (Child-TD)
        # 4,6 -> 1 (Child-ASD)
        # 1,3 -> 2 (Adult-TD)
        # 5,7 -> 3 (Adult-ASD)
        if k in [0, 2]: return 0
        if k in [4, 6]: return 1
        if k in [1, 3]: return 2
        if k in [5, 7]: return 3
        return -1
        
    preds_aq = np.array([map_to_age_quad(p) for p in preds])
    truth_aq = np.array([map_to_age_quad(t) for t in truth])
    
    acc_aq = accuracy_score(truth_aq, preds_aq)
    print(f"Model 2 (Age Quadruple) Accuracy:      {acc_aq*100:.2f}% (Paper: 85.42%)")

    # Generate Confusion Matrices
    plot_cm(truth, preds, "Octal Classification", ["M-Child-TD", "M-Adult-TD", "F-Child-TD", "F-Adult-TD", "M-Child-ASD", "M-Adult-ASD", "F-Child-ASD", "F-Adult-ASD"])
    plot_cm(truth_gq, preds_gq, "Gender Quadruple", ["M-TD", "M-ASD", "F-TD", "F-ASD"])
    plot_cm(truth_aq, preds_aq, "Age Quadruple", ["Child-TD", "Child-ASD", "Adult-TD", "Adult-ASD"])
    
def plot_cm(truth, preds, title, labels):
    cm = confusion_matrix(truth, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f"{title} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{title.replace(' ', '_')}_CM.png")
    print(f"Saved {title} Confusion Matrix.")

if __name__ == "__main__":
    # Update these paths!
    CKPT_PATH = "checkpoints/neuro-spectrum-epoch=49-val_acc_octal=0.68.ckpt" # Example, need to find actual latest
    ROOT_DIR = "data/nilearn/processed_3d"
    CSV_PATH = "data/nilearn/processed_3d/phenotypic.csv"
    
    # Find latest checkpoint automatically
    if not os.path.exists(CKPT_PATH):
        import glob
        list_of_files = glob.glob('checkpoints/*.ckpt') 
        if list_of_files:
            CKPT_PATH = max(list_of_files, key=os.path.getctime)
            print(f"Using latest checkpoint: {CKPT_PATH}")
        else:
            print("No checkpoints found!")
            exit()

    evaluate_paper_metrics(CKPT_PATH, ROOT_DIR, CSV_PATH)
