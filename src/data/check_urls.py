import requests
import logging

logging.basicConfig(level=logging.INFO)

def check_url(url):
    try:
        r = requests.head(url)
        if r.status_code == 200:
            print(f"[SUCCESS] Found: {url}")
            return True
        else:
            print(f"[FAILED] {r.status_code}: {url}")
    except Exception as e:
        print(f"[ERROR] {e}")
    return False

# Candidates to test
base = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"

# Sub ID details
# Site: PITT (Is it 'PITT', 'Pitt', 'pitt'?)
# Sub: 50003 (Is it '0050003'?)
sites = ['PITT', 'Pitt', 'NYU', 'Nyu', 'Yale', 'YALE']
sub_ids = ['0050003', '50003', '0050952', '50952'] # 50003 is Pitt, 50952 is NYU

pipelines = ['cpac', 'ants']
# Strategies: For ANATOMICAL, often there is NO strategy folder, or it is a dummy one?
# PCP docs say structural is processed using ANTs, CIVET, FreeSurfer.
# CPAC also outputs 'functional' derivatives?
strategies = ['filt_global', 'nofilt_noglobal', '']

derivatives = ['anat_mni', 'func_mean', 'anat_thickness'] 

for pipeline in pipelines:
    for strategy in strategies:
        for derivative in derivatives:
            for site in sites:
                for sub_id in sub_ids:
                    file_id = f"{site}_{sub_id}"
                    
                    # Pattern 1: With Strategy
                    if strategy:
                        url = f"{base}/{pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.nii.gz"
                        if check_url(url): break
                        
                    # Pattern 2: Without Strategy
                    url_root = f"{base}/{pipeline}/{derivative}/{file_id}_{derivative}.nii.gz"
                    if check_url(url_root): break

for pipeline in pipelines:
    for strategy in strategies:
        for site in sites:
            # Pattern 1: With Strategy
            file_id = f"{site}_{sub_id}"
            url = f"{base}/{pipeline}/{strategy}/anat_mni/{file_id}_anat_mni.nii.gz"
            check_url(url)
            
            # Pattern 2: Without Strategy (Maybe anatomical is root?)
            url_root = f"{base}/{pipeline}/anat_mni/{file_id}_anat_mni.nii.gz"
            check_url(url_root)
            
print("Done probing.")
