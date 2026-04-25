import torch
import torch.nn.functional as F
import numpy as np

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple of (gradient,)
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None, head='octal_logits'):
        """
        Generate Grad-CAM heatmap for a 3D volume.
        x: (B, C, D, H, W)
        """
        # Ensure model is in eval mode, but requires grads
        self.model.eval()
        x.requires_grad = True
        
        # Forward pass
        # model outputs dict, we need to extract the specific head
        outputs = self.model(x)
        logits = outputs[head]
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        
        # Get gradients and activations (B, C, D, H, W)
        gradients = self.gradients[0] # (C, D, H, W)
        activations = self.activations[0] # (C, D, H, W)
        
        # Global Average Pooling of gradients to get weights
        weights = torch.mean(gradients, dim=[1, 2, 3]) # (C)
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=x.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU to keep only positive influence
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max != cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        # Detach and convert to numpy
        cam = cam.cpu().detach().numpy()
        
        # Resize to match original input size using scipy or torch.nn.functional.interpolate
        # Input size is x.shape[2:]
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        return cam_resized.squeeze().numpy()
