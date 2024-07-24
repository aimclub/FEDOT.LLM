import zipfile
import json
import os
import pandas as pd
from scipy.io import arff

def unzip_archive(zip_name, extract_to):
    with zipfile.ZipFile(zip_name, 'r') as zipf:
        zipf.extractall(extract_to)

    print(f"{zip_name} extracted to {extract_to} successfully.")
        
def zip_archive(folder_path, zip_name):
    # Create a ZipFile object
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder and add files to the zip archive
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    print(f"{zip_name} created successfully.")