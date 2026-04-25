import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from model.system import NeuroSpectrumSystem

def test_inference():
    ckpt_path = "checkpoints/neuro-spectrum-epoch=18-val_acc_octal=1.00.ckpt"
    print(f"Loading model from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    try:
        model_system = NeuroSpectrumSystem.load_from_checkpoint(ckpt_path)
        model_system.eval()
        model_system.freeze()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Simulate processed input (128, 128, 128)
    print("Creating dummy input...")
    processed_vol = np.random.rand(128, 128, 128).astype(np.float32)
    processed_tensor = torch.from_numpy(processed_vol).float()
    
    # Add Batch and Channel dims: (1, 1, 128, 128, 128)
    input_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)
    print(f"Input tensor shape: {input_tensor.shape}")

    print("Running inference...")
    with torch.no_grad():
        outputs = model_system(input_tensor)
        logits = outputs['octal_logits']
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    print("Inference successful.")
    print(f"Probabilities: {probs}")
    print(f"Predicted Class Index: {np.argmax(probs)}")

if __name__ == "__main__":
    test_inference()
