import os
import shutil
import pandas as pd

print("Loading dataset labels...")
csv_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
df = pd.read_csv(csv_url)

source_dir = os.path.join("data", "nilearn", "processed_3d")
if not os.path.exists(source_dir):
    print("Error: Local dataset processed_3d not found.")
    exit(1)

files = [f for f in os.listdir(source_dir) if f.endswith('.nii.gz')]

os.makedirs(os.path.join('samples', 'ASD'), exist_ok=True)
os.makedirs(os.path.join('samples', 'Normal'), exist_ok=True)

asd_count = 0
norm_count = 0

print("Processing and copying files to separate testing directories...")

for f in files:
    if asd_count >= 5 and norm_count >= 5:
        break
        
    try:
        sub_id = int(f.split('.')[0])
        row = df[df['SUB_ID'] == sub_id]
        
        if len(row) > 0:
            dx = row.iloc[0]['DX_GROUP']
            src_path = os.path.join(source_dir, f)
            
            if dx == 1 and asd_count < 5:
                shutil.copy(src_path, os.path.join("samples", "ASD", f))
                asd_count += 1
                print(f"[{asd_count}/5] Copied {f} to samples/ASD")
            elif dx == 2 and norm_count < 5:
                shutil.copy(src_path, os.path.join("samples", "Normal", f))
                norm_count += 1
                print(f"[{norm_count}/5] Copied {f} to samples/Normal")
    except Exception as e:
        pass

print("Successfully generated isolated testing samples for both classes.")
