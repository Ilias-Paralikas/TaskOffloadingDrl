import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import json



def replace_keys(old_dict, key_mapping):
    return {key_mapping.get(k, k): v for k, v in old_dict.items()}


def get_last_averages(log_folder, key, n):
    all_averages = {}  # Initialize all_averages as a dictionary
    subfolders = [f.path for f in os.scandir(log_folder) if f.is_dir()]

    for subfolder in subfolders:
        averages = []  # Initialize averages as a list
        run_folders = os.listdir(subfolder)

        for run_folder in run_folders:
            metrics_file = os.path.join(subfolder,run_folder, 'metrics.pkl')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'rb') as f:
                    run_metrics = pickle.load(f)
                    if key in run_metrics:
                        # Get the last n values of the key
                        values = run_metrics[key][-n:]
                        average = np.mean([item for sublist in values for item in sublist])

                        averages.append(average)

        all_averages[os.path.basename(subfolder)] = np.array(averages)  # Store averages for the current subfolder

    return all_averages

def main():

    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
    parser.add_argument('--folder', type=str, default='meta_plots/logs/set_2/difficulties', help='path to the folder containing the logs')
    parser.add_argument('--plot_value', type=str, default ='rewards_history',help='name of the metric you want to plot. Note it must match the name in the metrics.pkl file')
    parser.add_argument('--average_window', type=int, default=500)
    args = parser.parse_args()  
    
    run_folder = os.path.join(args.folder ,'runs') 
    average_window = args.average_window  
    plot_value = args.plot_value  
    all_averages = get_last_averages(run_folder, plot_value, average_window)

    specifications_file = os.path.join(args.folder, 'specifications.json')
    with open(specifications_file, 'r') as f:
    # Load the JSON data from the file
        specifications = json.load(f)
        
    x_values = specifications['x_values']
    label_mapping = specifications['label_mapping']
    x_label = specifications['x_label']
    y_label  = specifications['y_label']
    df_column_name = specifications['df_column_name']
    if 'key_mapping' in specifications:
        key_mapping = specifications['key_mapping']
        all_averages = replace_keys(all_averages, key_mapping)
        
    # Create a new figure
    df = pd.DataFrame(all_averages, index=x_values)
    df.index.name = df_column_name
    df.to_csv(os.path.join(args.folder, plot_value+'.csv'))
    plt.rcParams['font.size'] = 17
    plt.figure(figsize=(8, 8))

    # Plot the averages for each subfolder
    for subfolder, averages in all_averages.items():
        plt.plot(x_values, averages, label=label_mapping.get(subfolder, subfolder))


    # plt.title('Baselines')  # Replace with your actual title


    plt.xlabel(x_label,fontsize=25)  # Replace with your actual x axis name
    plt.ylabel(y_label,fontsize=25)
    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(os.path.join(args.folder, plot_value+'.png'),bbox_inches='tight',dpi=500)


if __name__ == '__main__':
    main()