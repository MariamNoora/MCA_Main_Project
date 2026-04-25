import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any
from .resnet3d_multiclass import NeuroSpectrumNet

class NeuroSpectrumSystem(L.LightningModule):
    """
    Lightning System for NeuroSpectrum AI.
    Handles Multi-Head Training (Age, Gender, Octal).
    """
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuroSpectrumNet()
        self.lr = lr
        
        # Loss functions (Standard Cross Entropy)
        # We could add class weights here if we calculate them from dataset
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, outputs, batch):
        """
        Computes weighted loss for all heads.
        """
        # Targets
        gender_target = batch['sex']
        age_target = batch['age']
        octal_target = batch['octal'] # 8 classes
        
        # Predictions
        gender_logits = outputs['gender_logits']
        age_logits = outputs['age_logits']
        octal_logits = outputs['octal_logits']
        
        # Losses
        loss_gender = self.criterion(gender_logits, gender_target)
        loss_age = self.criterion(age_logits, age_target)
        loss_octal = self.criterion(octal_logits, octal_target)
        
        # Total Loss (Simple Sum or Weighted)
        # Benchmarks show joint training helps.
        loss_total = loss_gender + loss_age + loss_octal
        
        return loss_total, {'loss_gender': loss_gender, 'loss_age': loss_age, 'loss_octal': loss_octal}

    def training_step(self, batch, batch_idx):
        x, _ = batch # Dataset returns (tensor, labels_dict)
        # But wait, dataset returns (tensor, dict).
        # Pytorch DataLoader collates (list_of_tensors, list_of_dicts)?
        # Default collate might fail on list of dicts unless we handle it.
        # Actually standard collate for dicts stacks them if values are numeric.
        # My dataset returns `final_tensor, {'dx': ..., 'sex': ...}`
        # The dict values are scalars (int/float).
        # Check Dataset: `label_dx` is int.
        # So batch is `[params_tensor, dict_of_stacked_tensors]`. (Using tuple unpacking `x, labels = batch`)
        
        x, labels = batch
        
        outputs = self(x)
        loss, loss_dict = self._calculate_loss(outputs, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(loss_dict)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        outputs = self(x)
        loss, loss_dict = self._calculate_loss(outputs, labels)
        
        # Calculate Accuracies
        acc_gender = (outputs['gender_logits'].argmax(1) == labels['sex']).float().mean()
        acc_age = (outputs['age_logits'].argmax(1) == labels['age']).float().mean()
        acc_octal = (outputs['octal_logits'].argmax(1) == labels['octal']).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_octal', acc_octal, prog_bar=True)
        self.log('val_acc_gender', acc_gender)
        self.log('val_acc_age', acc_age)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
