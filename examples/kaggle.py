import os
import kaggle
import zipfile
import time
from datetime import datetime

def download_from_kaggle(competition_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading dataset to: {save_path}")
    kaggle.api.competition_download_files(competition_name, path=save_path)
    
    # Extract main zip file
    zip_file_path = os.path.join(save_path, f'{competition_name}.zip')
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    os.remove(zip_file_path)
    
    # Extract any nested zip files
    for root, _, files in os.walk(save_path):
        for file in files:
            if file.lower().endswith('.zip'):
                extract_dir = os.path.splitext(root)[0]
                try:
                    print(f"Extracting nested zip: {os.path.join(root, file)}")
                    with zipfile.ZipFile(os.path.join(root, file), 'r') as nested_zip:
                        nested_zip.extractall(extract_dir)
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    print(f"Error extracting {os.path.join(root, file)}: {e}")
    
    print(f"Dataset downloaded and extracted to {save_path}")

def submit_to_kaggle(competition_name, submission_file, submission_message=None):
    """Submit a file to Kaggle competition and return submission details."""
    if not submission_message:
        submission_message = f"FedotLLM_submission_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    if not os.path.exists(submission_file):
        print(f"Submission file not found at {submission_file}")
        return None
    
    try:
        print(f"Submitting to '{competition_name}'...")
        kaggle.api.competition_submit(submission_file, submission_message, competition_name)
        while True:
            submissions = kaggle.api.competition_submissions(competition_name)
            if not submissions:
                print("No submissions found.")
                return None
                
            latest = submissions[0]
            if str(latest.status).lower() != 'pending' and str(latest.status).lower() != 'error' and latest.public_score:
                    if hasattr(latest, 'error_description') and 'scoring' in str(latest.error_description).lower():
                        pass # Still scoring, continue loop
                    else:
                        break # Scoring complete or final state reached
            
            print(f"Current status: {latest.status} (Description: {getattr(latest, 'error_description', 'N/A')}). Waiting for scores...")
            time.sleep(30) 
        print(f"\nSubmission details:\nDate: {latest.date}\nStatus: {latest.status}\n"
              f"Public Score: {latest.public_score}\n"
              f"Private Score: {getattr(latest, 'private_score', 'Not available')}")
        return latest
        
    except Exception as e:
        print(f"Error submitting to Kaggle: {e}")
        return None