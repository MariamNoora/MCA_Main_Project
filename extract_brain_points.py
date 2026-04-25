import numpy as np
from nilearn import datasets
from nilearn.surface import load_surf_mesh
import json
import os

try:
    print("Fetching fsaverage surface...")
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    
    print("Loading left and right hemispheres...")
    coords_l, _ = load_surf_mesh(fsaverage['pial_left'])
    coords_r, _ = load_surf_mesh(fsaverage['pial_right'])
    
    coords = np.vstack((coords_l, coords_r))
    
    # Subsample 15,000 points
    print(f"Total points: {coords.shape[0]}. Sampling 15,000...")
    np.random.seed(42)
    indices = np.random.choice(coords.shape[0], 15000, replace=False)
    sampled = coords[indices]
    
    # Center and normalize scale
    center = np.mean(sampled, axis=0)
    sampled -= center
    
    # Scale to roughly 2.0 width
    max_val = np.max(np.abs(sampled))
    sampled /= (max_val / 2.0)
    
    # Rotate 90 degrees around X to face forward in Three.js (usually MNI coordinates need rotation)
    # MNI has Z as superior, Y as anterior, X as right.
    # ThreeJS has Y as up, Z as forward, X as right.
    # So Y_three = Z_mni, Z_three = -Y_mni, X_three = X_mni
    
    three_coords = []
    for point in sampled:
        x, y, z = point
        three_coords.extend([float(x), float(z), float(-y)])
        
    output_path = os.path.join('frontend', 'public', 'brain_points.json')
    with open(output_path, 'w') as f:
        json.dump(three_coords, f)
        
    print(f"Successfully saved {len(three_coords)//3} points to {output_path}")

except Exception as e:
    print(f"Error: {e}")
