import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument('--folder', type=str, default='meta_plots/logs/set_2/action_comparison', help='path to the folder containing the logs')
    parser.add_argument('--average_window', type=int, default=500)
    args = parser.parse_args()  
    
    run_folder = os.path.join(args.folder ,'runs') 
    average_window = args.average_window  

    specifications_file = os.path.join(args.folder, 'specifications.json')
    with open(specifications_file, 'r') as f:
    # Load the JSON data from the file
        specifications = json.load(f)
    
    metrics=  load_all_metrics(run_folder)

    sums = {key:{'local': 0, 'horisontal': 0, 'cloud': 0} for key in metrics.keys()}
    for run in metrics:
        actions_history = metrics[run]['actions_history'][-average_window:]  # Get last 500 or fewer elements
        
        # Initialize sums
        for total_actions in actions_history:
            # Check if action is a dictionary
            for action in total_actions:
                for key in sums[run].keys():
                    sums[run][key] += action[key]/args.average_window

    # Assuming 'sums' is your nested dictionary and 'categories' is defined
    categories = ['local', 'horisontal', 'cloud']
    dict_keys = list(sums.keys())  # Get the keys from your dictionary
    num_dicts = len(sums)
    colors = plt.cm.viridis(np.linspace(0, 1, num_dicts))  # Generate colors
    fig, ax = plt.subplots()

    # Number of categories
    num_categories = len(categories)

    # Calculate the total width for all bars in one group
    total_bar_width = 0.8
    single_bar_width = total_bar_width / num_dicts

    # Calculate the offset to center the bars
    offset = total_bar_width / num_dicts / 2

    # Iterate over each category to plot
    for i, category in enumerate(categories):
        # Extract values for this category from each dict and plot them
        for j, key in enumerate(dict_keys):
            value = sums[key][category]
            # Calculate the position for each bar
            position = i - offset + (j + 0.5) * single_bar_width
            ax.bar(position, value, color=colors[j], width=single_bar_width, label=key if i == 0 and j == 0 else "")

    # Set the position and labels for the x-ticks
    ax.set_xticks(range(num_categories))
    ax.set_xticklabels(categories)

    # Adding legend and labels
    mapped_values = [specifications['label_mapping'][key] for key in dict_keys if key in specifications['label_mapping']]
    ax.legend(mapped_values, title=specifications['legend_title'])
    ax.set_ylabel('Values')
    ax.set_title(specifications['title'])

    plt.savefig(os.path.join(args.folder, 'actions_comparison.png'), dpi=500)
    
if __name__ == "__main__":
    main()