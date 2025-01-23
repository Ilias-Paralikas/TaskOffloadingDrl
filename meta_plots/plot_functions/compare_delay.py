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
    parser.add_argument('--folder', type=str, default='meta_plots/logs/comparissons', help='path to the folder containing the logs')
    parser.add_argument('--average_window', type=int, default=50)
    args = parser.parse_args()
    fig, axs = plt.subplots(3, 3, figsize=(24, 24))
    plt.rcParams['font.size'] = 17
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plot_idx = 0

    for f in ['medium', 'hard', 'very_hard']:
        for plot_value in ['task_drop_ratio_history', 'energy_rewards_history', 'delay_without_drop_rewards_history']:
            ax = axs[plot_idx // 3, plot_idx % 3]
            plot_idx += 1

            folder = os.path.join(args.folder, f)
            run_folder = os.path.join(folder, 'runs')
            average_window = args.average_window
            all_averages = get_last_averages(run_folder, plot_value, average_window)

            if plot_value == 'delay_without_drop_rewards_history':
                task_arrive = 100 * 0.7
                task_drop_ratio = get_last_averages(run_folder, 'task_drop_ratio_history', average_window)
                tasks_completed = {k: task_arrive * (1 - e) for k, e in task_drop_ratio.items()}
                total_delay = get_last_averages(run_folder, 'delay_without_drop_rewards_history', average_window)
                total_delay = {k: -e for k, e in total_delay.items()}
                all_averages = {k: e / tasks_completed[k] for k, e in total_delay.items()}

            if plot_value == 'energy_rewards_history':
                all_averages = {k: -e for k, e in all_averages.items()}

            y_label_dict = {
                'rewards_history': 'Reward',
                'task_drop_ratio_history': 'Task Drop Ratio',
                'energy_rewards_history': 'Energy Consumption',
                'delay_without_drop_rewards_history': 'Delay'
            }

            specifications_file = os.path.join(folder, 'specifications.json')
            with open(specifications_file, 'r') as specs:
                specifications = json.load(specs)

            x_values = specifications['x_values']
            label_mapping = specifications['label_mapping']
            x_label = specifications['x_label']
            y_label = y_label_dict[plot_value]
            df_column_name = 'column_name'

            if 'key_mapping' in specifications:
                key_mapping = specifications['key_mapping']
                all_averages = replace_keys(all_averages, key_mapping)

            df = pd.DataFrame(all_averages, index=x_values)
            df.index.name = df_column_name
            df.to_csv(os.path.join(folder, plot_value + '.csv'))

            markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>', '1', '2', '3', '4', 'h', 'H', '|', '_']

            for (subfolder, averages), marker in zip(all_averages.items(), markers):
                ax.plot(x_values, averages, label=label_mapping.get(subfolder, subfolder), marker=marker)

            ax.set_xlabel(x_label, fontsize=25)
            ax.set_ylabel(y_label, fontsize=25)
            ax.legend(title=specifications['legend_title'])

    plt.savefig(os.path.join(args.folder, 'combined_plots.png'), bbox_inches='tight', dpi=500)


if __name__ == '__main__':
    main()