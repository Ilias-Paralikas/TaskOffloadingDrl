import json
from deepdiff import DeepDiff
import os 

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_json_files(file1, file2):
    json1 = load_json(file1)
    json2 = load_json(file2)
    
    differences = DeepDiff(json1, json2, ignore_order=True)
    return differences

if __name__ == "__main__":

    file1 = 'meta_plots/logs/decoffe/learning_rate/runs/run_0/hyperparameters.json'
    file2 = 'meta_plots/logs/decoffe/gamma/runs/run_0/hyperparameters.json'
    
    differences = compare_json_files(file1, file2)
    
    if differences:
        print("Differences found:")
        print(json.dumps(differences, indent=4))
    else:
        print("No differences found.")