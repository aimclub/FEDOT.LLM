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

def load_dataset_data(path):
    #load metadata file
    with open(os.sep.join([path, 'metadata.json']), 'r') as json_file:
        dataset_metadata = json.load(json_file)

    #load each split file
    dataset_metadata["splits"] = {}
    for split_name in dataset_metadata['split_names']:
        split_path = os.sep.join([path, dataset_metadata["split_paths"][split_name]]) 
        if split_path.split(".")[-1] == "csv":
            dataset_metadata["splits"][split_name] = pd.read_csv(split_path)
        elif split_path.split(".")[-1] == "arff":
            data = arff.loadarff(split_path)
            dataset_metadata["splits"][split_name] = pd.DataFrame(data[0])
        else:
            print(f"split {split_path}: unsupported format")
                  
    #if we have model responses saved already
    if os.path.exists(os.sep.join([path,'model_responses.json'])):
        with open(os.sep.join([path, 'model_responses.json']), 'r') as json_file:
            model_responses = json.load(json_file)
            dataset_metadata.update(model_responses)
            
    return dataset_metadata

def print_dataset_data(dataset_metadata):
    for key in dataset_metadata:
      if key != 'splits':
        print(key, ":", dataset_metadata[key])
      else:
        print(key, ":", dataset_metadata[key].keys())