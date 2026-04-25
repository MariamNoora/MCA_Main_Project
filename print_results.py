
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.model.system import NeuroSpectrumSystem
from src.data.dataset import NeuroSpectrumDataset
from sklearn.metrics import accuracy_score

def eval_simple():
    # Find latest checkpoint
    import glob
    list_of_files = glob.glob('checkpoints/*.ckpt') 
    if list_of_files:
        ckpt_path = max(list_of_files, key=os.path.getctime)
    else:
        print("No Checkpoint")
        return

    system = NeuroSpectrumSystem.load_from_checkpoint(ckpt_path)
    system.eval()
    system.freeze()
    
    dataset = NeuroSpectrumDataset(root_dir="data/nilearn/processed_3d", csv_path="data/nilearn/processed_3d/phenotypic.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    all_preds_octal = []
    all_labels_octal = []
    
    print("Inference Start")
    for batch in dataloader:
        x, labels = batch
        outputs = system(x)
        pred_octal = torch.softmax(outputs['octal_logits'], dim=1).argmax(1).cpu().numpy()
        all_preds_octal.extend(pred_octal)
        all_labels_octal.extend(labels['octal'].cpu().numpy())
        
    preds = np.array(all_preds_octal)
    truth = np.array(all_labels_octal)
    
    # Octal
    acc_octal = accuracy_score(truth, preds)
    
    # Gender Quad
    def map_to_gender_quad(k):
        if k in [0, 1]: return 0
        if k in [4, 5]: return 1
        if k in [2, 3]: return 2
        if k in [6, 7]: return 3
        return -1
    preds_gq = np.array([map_to_gender_quad(p) for p in preds])
    truth_gq = np.array([map_to_gender_quad(t) for t in truth])
    acc_gq = accuracy_score(truth_gq, preds_gq)
    
    # Age Quad
    def map_to_age_quad(k):
        if k in [0, 2]: return 0
        if k in [4, 6]: return 1
        if k in [1, 3]: return 2
        if k in [5, 7]: return 3
        return -1
    preds_aq = np.array([map_to_age_quad(p) for p in preds])
    truth_aq = np.array([map_to_age_quad(t) for t in truth])
    acc_aq = accuracy_score(truth_aq, preds_aq)
    
    print(f"FINAL_METRICS: Octal={acc_octal:.4f}, GenderQuad={acc_gq:.4f}, AgeQuad={acc_aq:.4f}")

if __name__ == "__main__":
    eval_simple()
