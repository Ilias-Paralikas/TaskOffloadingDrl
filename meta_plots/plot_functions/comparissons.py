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
    """
    Generates comparative plots for different metrics across multiple experimental runs.
    This function processes experimental data and creates plots comparing different baselines
    or configurations based on specified metrics. It handles two types of metrics by default:
    task drop ratio and energy rewards history.
    The function performs the following main operations:
    1. Processes command line arguments for configuration
    2. Reads experimental data from specified folders
    3. Processes the data according to the metric type
    4. Creates and saves both plots and CSV files of the processed data
    Args:
        None (uses command line arguments)
    Command Line Arguments:
        --folder (str): Path to the folder containing the logs and specifications
                       Default: 'meta_plots/logs/comparissons/medium'
        --plot_value (str): Metric to plot (must match metrics.pkl file name)
                           Default: 'task_drop_ratio_history'
        --average_window (int): Window size for averaging the results
                               Default: 20
    Required Files in Folder:
        - specifications.json: Contains plotting specifications including:
            - x_values: Values for x-axis
            - label_mapping: Dictionary mapping raw labels to display labels
            - x_label: Label for x-axis
            - legend_title: Title for the plot legend
            - key_mapping (optional): Dictionary for mapping keys to new values
    Output:
        - PNG plot file: Saved as {plot_value}.png
        - CSV data file: Saved as {plot_value}.csv
    Note:
        - Supports automatic inversion of energy reward values
        - Uses different markers for each plotted line
        - Customizes plot appearance (font size, figure size)
        - Handles multiple experimental runs and configurations
    """
    for plot_value in ['task_drop_ratio_history', 'energy_rewards_history']:
        parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
        parser.add_argument('--folder', type=str, default='meta_plots/logs/comparissons/medium', help='path to the folder containing the logs')
        parser.add_argument('--plot_value', type=str, default ='task_drop_ratio_history',help='name of the metric you want to plot. Note it must match the name in the metrics.pkl file')
        parser.add_argument('--average_window', type=int, default=20)
        args = parser.parse_args()
        run_folder = os.path.join(args.folder ,'runs') 
        average_window = args.average_window  
        # plot_value = args.plot_value  
        all_averages = get_last_averages(run_folder, plot_value, average_window)
        if plot_value =='energy_rewards_history':
            all_averages = {k:-e for k,e in all_averages.items()}
        y_label_dict ={'rewards_history':'Reward',
                    'task_drop_ratio_history':'Task Drop Ratio',
                    'energy_rewards_history':'Energy Consumption'}


        specifications_file = os.path.join(args.folder, 'specifications.json')
        with open(specifications_file, 'r') as f:
        # Load the JSON data from the file
            specifications = json.load(f)
            
            
        x_values = specifications['x_values']
        label_mapping = specifications['label_mapping']
        x_label = specifications['x_label']

        y_label = y_label_dict[plot_value]    
        df_column_name = 'column_name'
        if 'key_mapping' in specifications:
            key_mapping = specifications['key_mapping']
            all_averages = replace_keys(all_averages, key_mapping)
            
        # Create a new figure
        df = pd.DataFrame(all_averages, index=x_values)
        df.index.name = df_column_name
        df.to_csv(os.path.join(args.folder, plot_value+'.csv'))
        plt.rcParams['font.size'] = 17
        plt.figure(figsize=(8, 8))

        markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>', '1', '2', '3', '4', 'h', 'H', '|', '_']

        # Plot the averages for each subfolder
        for (subfolder, averages), marker in zip(all_averages.items(), markers):
            plt.plot(x_values, averages, label=label_mapping.get(subfolder, subfolder), marker=marker)

        # plt.title('Baselines')  # 

        # plt.title('Baselines')  # Replace with your actual title


        plt.xlabel(x_label,fontsize=25)  # Replace with your actual x axis name
        plt.ylabel(y_label,fontsize=25)
        # Add a legend
        plt.legend( title=specifications['legend_title'])

        # Show the plot
        plt.savefig(os.path.join(args.folder, plot_value+'.png'),bbox_inches='tight',dpi=500)


if __name__ == '__main__':
    main()