import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import argparse
import json


def load_all_metrics(log_folder):
    metrics = {}
    run_folders = os.listdir(log_folder)

    for run_folder in run_folders:
        metrics_file = os.path.join(log_folder,run_folder, 'metrics.pkl')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'rb') as f:
                metrics[os.path.basename(run_folder)] = pickle.load(f)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')

    parser.add_argument('--plot_value', type=str, default ='rewards_history',help='name of the metric you want to plot. Note it must match the name in the metrics.pkl file')
    parser.add_argument('--folder', type=str, default='meta_plots/logs/set_2/gamma', help='path to the folder containing the logs')
    parser.add_argument('--average_window', type=int, default=2000)
    parser.add_argument('--clip', type=int, default=5000)
    parser.add_argument('--marker_every', type=int, default=200)


    args = parser.parse_args()  # Parse the command line arguments
    plot_value = args.plot_value  # Get the plot_value from the command line arguments
    folder = os.path.join(args.folder ,'runs') # Get the folder from the command line arguments
    average_window = args.average_window  # Set the size of the moving average window

    metrics =  load_all_metrics(folder)

    values = {run: data[plot_value] for run, data in metrics.items() if plot_value in data}
    if args.clip:
        values = {run: data[:args.clip] for run, data in values.items()}

    # Calculate moving average of rewards history for each run
    average_values = {run: [np.mean(rewards[max(0, i - average_window):i]) for i in range(1, len(rewards))] for run, rewards in values.items()}
    average_values =  dict(sorted(average_values.items()))
    
    min_length = min(len(v) for v in average_values.values())
    specifications_file = os.path.join(args.folder, 'specifications.json')
    with open(specifications_file, 'r') as f:
    # Load the JSON data from the file
        specifications = json.load(f)
# Define a list of marker styles
    markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>', '1', '2', '3', '4', 'h', 'H', '|', '_']

    # Ensure the markers list is at least as long as the number of keys in average_values
    if len(markers) < len(average_values):
        print("Warning: Not enough unique markers for each plot. Consider adding more markers.")
        # Optionally, repeat the markers list to cover all plots
        markers *= (len(average_values) // len(markers)) + 1

    # Plot lines with different markers
    for (key, marker) in zip(average_values, markers):
        label = specifications['label_mapping'].get(key, key)  # Use key as default if not found in label_mapping

        plt.plot(average_values[key], label=label, marker=marker, markevery=args.marker_every)

        
    plt.xlabel('episode')
    plt.ylabel(specifications['y_label'])
    plt.legend()
    plt.savefig(os.path.join(folder, args.plot_value+'.png'),dpi=500)

    
    # Truncate all lists to the same length
    average_values = {k: v[:min_length] for k, v in average_values.items()}

    df = pd.DataFrame(average_values)
    df.to_csv(os.path.join(folder, args.plot_value+'.csv'), index=False)
    
if __name__ == '__main__':
    main()