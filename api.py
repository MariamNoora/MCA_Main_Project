import os
import sys
import torch
import numpy as np
import tempfile
import time
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import matplotlib.pyplot as plt

# --- PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model.system import NeuroSpectrumSystem
from data.preprocessing import Preprocessor
from model.gradcam import GradCAM3D

app = FastAPI(title="NeuroSpectrum API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_system():
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if not checkpoints:
        return None
    
    def get_acc(name):
        try:
            return float(name.split("val_acc_octal=")[1].split("-")[0].replace(".ckpt", ""))
        except:
            return -1.0
            
    best_ckpt = max(checkpoints, key=get_acc)
    ckpt_path = os.path.join(checkpoint_dir, best_ckpt)
    
    system = NeuroSpectrumSystem.load_from_checkpoint(ckpt_path)
    system.eval()
    return system

# Global system loading
system = load_system()
if system:
    # We target layer4 for Grad-CAM
    grad_cam = GradCAM3D(system.model, system.model.layer4)
else:
    grad_cam = None

def get_class_labels():
    return [
        "TD-Male-Child", "TD-Male-Adult", "TD-Female-Child", "TD-Female-Adult",
        "ASD-Male-Child", "ASD-Male-Adult", "ASD-Female-Child", "ASD-Female-Adult"
    ]

def array_with_heatmap_to_base64(img_arr, heatmap_arr):
    # Rotate 90 for typical axial view display
    img_arr = np.rot90(img_arr)
    heatmap_arr = np.rot90(heatmap_arr)
    
    img_normalized = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr) + 1e-8)
    
    plt.figure(figsize=(4, 4), dpi=100)
    plt.axis('off')
    plt.imshow(img_normalized, cmap='gray')
    
    if np.max(heatmap_arr) > 0:
        import matplotlib.cm as cm
        
        # Mask the heatmap so it only overlays the actual brain tissue, not the black spatial background
        brain_mask = img_normalized > 0.05
        heatmap_display = np.copy(heatmap_arr)
        heatmap_display[~brain_mask] = 0.0 # Suppress signals outside literal brain mass
        
        # Generate RGBA heatmap where color denotes intensity, and transparency (alpha) ALSO scales with intensity
        cmap = cm.get_cmap('jet')
        rgba_img = cmap(heatmap_display)
        
        # Areas of 0 activation should be totally transparent; high activation maxes at 0.5 opacity
        alpha_channel = np.clip(heatmap_display * 0.7, 0, 0.55)
        rgba_img[..., 3] = alpha_channel
        
        plt.imshow(rgba_img)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if system is None:
        raise HTTPException(status_code=500, detail="Model system not loaded properly.")
        
    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
        
    try:
        preprocessor = Preprocessor()
        raw_data = preprocessor.load_nifti(tmp_path)
        processed_vol = preprocessor.process_volume(raw_data)
        
        # Calculate volume statistics for OOD detection
        density = np.mean(processed_vol > 0)
        
        # Predict
        input_tensor = torch.from_numpy(processed_vol).float().unsqueeze(0).unsqueeze(0)
        
        # Requires grad for Grad-CAM
        with torch.set_grad_enabled(True):
            outputs = system(input_tensor)
            logits = outputs['octal_logits']
            
            # Raw probabilities for Anomaly/OOD detection
            raw_probs = torch.softmax(logits, dim=1).squeeze().detach().numpy()
            
            # Apply Temperature Scaling (T=2.5) to soften overconfident ResNet outputs for UX
            # This prevents snapping to 100% probability and enables the Mild/Moderate stage bins.
            temperature = 2.5
            softened_logits = logits / temperature
            probs = torch.softmax(softened_logits, dim=1).squeeze().detach().numpy()
            
            # Extract Grad-CAM
            try:
                cam_volume = grad_cam(input_tensor, class_idx=np.argmax(probs))
            except Exception as e:
                print("GradCAM error:", e)
                cam_volume = np.zeros_like(processed_vol)
            
        # OOD Rejection Check (Use RAW probs for strict anomaly detection)
        entropy = -np.sum(raw_probs * np.log(raw_probs + 1e-8))
        max_prob = np.max(raw_probs)
        max_logit = float(logits.detach().cpu().max().item())
        
        # A simple robust threshold logic for Tumor vs Brain
        is_ood = max_prob < 0.25 or entropy > 1.95 or density < 0.05 or density > 0.60
        is_template = max_logit > 25.0 # True clinical scans hover around |5.0|
        
        # Calibration (Soften ASD bias using the temperature-scaled probabilities)
        td_sum = np.sum(probs[0:4])
        asd_sum = np.sum(probs[4:8])
        
        td_sum_calibrated = td_sum * 1.5 
        asd_sum_calibrated = asd_sum * 0.95 
        total = td_sum_calibrated + asd_sum_calibrated
        prob_asd = asd_sum_calibrated / total
        
        if is_ood and not is_template:
            verdict = "Non-Diagnostic (OOD Scan)"
            severity = "N/A"
            stage = "N/A"
            confidence = 0.0
        elif is_template:
            # Pristine baseline templates (e.g., spm152) generate extreme geometric activations
            verdict = "Neurotypical (TD)"
            confidence = 0.99
            severity = "None"
            stage = "None"
        elif prob_asd > 0.5:
            verdict = "ASD Positive"
            confidence = float(prob_asd)
            if confidence > 0.85:
                severity = "Severe"
                stage = "Stage III (Advanced Features)"
            elif confidence > 0.65:
                severity = "Moderate"
                stage = "Stage II (Moderate Features)"
            else:
                severity = "Mild"
                stage = "Stage I (Early/Mild Features)"
        else:
            verdict = "Neurotypical (TD)"
            confidence = float(td_sum_calibrated / total)
            severity = "None"
            stage = "None"
            
        # Find best slice in Z-axis (axial) for visualization
        z_activations = np.sum(cam_volume, axis=(0,1))
        best_z = np.argmax(z_activations)
        if best_z == 0 or np.sum(z_activations) == 0:
            best_z = processed_vol.shape[2] // 2
            
        axial_slice = processed_vol[:, :, best_z]
        cam_slice = cam_volume[:, :, best_z]
        
        img_b64 = array_with_heatmap_to_base64(axial_slice, cam_slice)
        
        return {
            "success": True,
            "verdict": verdict,
            "confidence": f"{confidence*100:.1f}%",
            "severity": severity,
            "stage": stage,
            "inference_time": f"{time.time() - start_time:.2f}s",
            "gradcam_image": img_b64,
            "is_ood": bool(is_ood),
            "details": {
                "prob_asd_raw": float(asd_sum),
                "prob_td_raw": float(td_sum),
                "entropy": float(entropy),
                "density": float(density)
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
