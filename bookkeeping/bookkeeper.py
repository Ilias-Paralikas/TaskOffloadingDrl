import os 
import json
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import sum_dicts_in_positions
from topology_generators import plot_matrix
from lr_schedulers import *



class BookKeeper:
    """
    The BookKeeper class is designed to manage and store all relevant data, metrics, 
    and configurations required during a deep reinforcement learning (DRL) process for task offloading. 
    It serves as a central utility for creating and maintaining logs, saving hyperparameters, 
    tracking training progress, and generating various plots to visualize performance metrics and actions.
    Key Responsibilities:
    ---------------------
    1. Folder and File Management:
        - Creates and organizes folders to store logs, checkpoints, metrics, and configuration files.
        - Manages run indices to ensure each new training session is stored in a uniquely numbered folder.
        - Reads hyperparameters from a specified JSON file, saves them in the newly created run folder, 
          and optionally loads from an existing run folder when resuming.
    2. Tracking and Storage of Metrics:
        - Keeps a running record of rewards, dropped tasks, arrived tasks, and other performance metrics during training.
        - Maintains metrics history across episodes in a serialized (pickle) file, allowing for pause and resume functionality.
        - Stores epsilon history to keep track of the exploration rate in the RL algorithm.
    3. Scheduler Management:
        - Initializes and stores the chosen learning rate scheduler, such as a linear or constant decay.
        - Allows retrieval of the stored scheduler when resuming an experiment.
    4. Plot Generation:
        - Generates PNG plots for various metrics (e.g., rewards, delay without drop, energy rewards, 
          task drop ratio) and computes moving averages for smoother trend visualization.
        - Creates separate action plots illustrating how many times each agent chose a local, 
          horizontal (typo-horisontal), or cloud approach.
    5. Customization and Extensibility:
        - Permits the addition of new metrics in the "metrics" dictionary, facilitating easy extension of 
          logging without modifying the existing core code structure.
        - Provides methods for retrieving important file paths (e.g., checkpoints, scheduler, run folder) 
          to simplify organization in larger codebases.
    Arguments in __init__:
    ----------------------
    • log_folder (str): 
         The path where all logs and run-specific files will be saved. Defaults to 'log_folder'.
    • hyperparameters_source_file (str or None): 
         The JSON file containing hyperparameters. If 'resume_run' is not specified, 
         the contents of this file are loaded and saved into the run folder.
    • resume_run (str or None): 
         The name of an existing log folder run to resume training from. 
         If provided, metrics and hyperparameters are loaded from this folder.
    • average_window (int): 
         Determines how many of the most recent episodes to include in the moving average 
         calculation for metrics.
    Class Attributes:
    ----------------
    • plotable_metrics (list of str): 
         A list defining which metric keys should be plotted.
    • run_folder (str): 
         The folder that corresponds to the current run; populated with folder hierarchy such as 
         checkpoints, metrics, and hyperparameters.
    • checkpoint_folder (str): 
         Directory where checkpoint files (policy, NN weights) are stored.
    • metrics_folder (str): 
         File path pointing to the pickle file that stores the training metrics.
    • scheduler_file (str): 
         File path pointing to the pickle file that stores the learning rate scheduler.
    • metrics (dict): 
         A dictionary that accumulates and stores all the relevant performance data 
         (e.g., rewards, task drop ratio, epsilon history, etc.).
    • delay_without_drop_rewards, energy_rewards, rewards, tasks_dropped, tasks_arrived (lists): 
         Lists that hold step-level metrics, which get aggregated and appended to "metrics" at the end of each episode.
    Key Methods:
    ------------
    • get_epsilon():
         Retrieves the latest epsilon (exploration rate) value from the metrics.
    • get_checkpoint_folder():
         Returns the path to the directory where checkpoint files are stored.
    • get_scheduler_file():
         Provides the path to the stored learning rate scheduler pickle file.
    • get_hyperparameters():
         Opens the hyperparameters file and returns its contents, including the current epsilon value.
    • store_step(info):
         Accumulates data from each step of an episode into local lists before final aggregation.
    • store_episode(epsilon, actions):
         Performs the episode-level aggregation of metrics. Summarizes step-level data into 
         single-episode statistics (sums of rewards, tasks, etc.). Saves the updated metrics to a pickle file 
         and resets step-level lists for the next episode.
    • plot_and_save(key):
         Generates a line plot of a chosen metric for each agent alongside their mean, 
         then saves the figure in the run folder.
    • moving_average(a):
         Computes a moving average over the last 'average_window' episodes for a given 1D array.
    • plot_and_save_moving_avg(key):
         Similar to plot_and_save but uses the smoothed data derived from "moving_average" for each agent 
         and a mean line for clarity.
    • plot_actions():
         Creates plots that represent how many times each agent chose local, horizontal, or cloud actions 
         during the episodes. Also includes a total-aggregated plot.
    • plot_metrics():
         Iterates over all metrics in the "metrics" dictionary and generates both 
         regular and moving average plots (if they are in "plotable_metrics"), 
         then calls "plot_actions()" to plot action distributions.
    • get_run_folder():
         Retrieves the path of the currently active run folder.
    • get_rewards_history():
         Returns the entire history of rewards stored in "metrics['rewards_history']".
    Usage Scenario:
    ---------------
    • Instantiate BookKeeper by specifying a log folder and hyperparameters source file 
      or by referencing a previously created run folder to resume.
    • After each environment step, call store_step() with the relevant reward and action data.
    • After each episode, call store_episode() to finalize and log the metrics for that episode.
    • Use plot_metrics() to visualize trends, monitor performance, and guide further 
      training or hyperparameter tuning.
    """
    def __init__(self, log_folder='log_folder', hyperparameters_source_file=None, resume_run=None, average_window=500):
        self.plotable_metrics = ['rewards_history', 'delay_without_drop_rewards_history', 'energy_rewards_history', 'task_drop_ratio_history']
        self.log_folder = log_folder
        self.average_window = average_window
        os.makedirs(log_folder, exist_ok=True)

        if resume_run:
            # If resuming a run, set the run folder and hyperparameters file
            self.run_folder = os.path.join(log_folder, resume_run)
            self.hyperparameters_file = os.path.join(self.run_folder, 'hyperparameters.json')
        else:
            # Load hyperparameters from the source file
            with open(hyperparameters_source_file) as f:
                hyperparameters = json.load(f)
            
            # Manage run index
            index_filepath = log_folder + '/index.txt'
            if not os.path.exists(index_filepath):
                with open(index_filepath, 'w') as file:
                    file.write('0')
                    run_index = 0
            else:
                with open(index_filepath, 'r') as file:
                    run_index = int(file.read().strip())
                run_index += 1
                with open(index_filepath, 'w') as file:
                    file.write(str(run_index))
            
            # Create run folder and save hyperparameters
            self.run_folder = log_folder + '/run_' + str(run_index)
            os.makedirs(self.run_folder, exist_ok=True)
            self.hyperparameters_file = self.run_folder + '/hyperparameters.json'
            json_object = json.dumps(hyperparameters, indent=4)
            with open(self.hyperparameters_file, "w") as outfile:
                outfile.write(json_object)
            
            # Plot and save the connection matrix
            connection_matrix = hyperparameters['connection_matrix']
            topology_target_file = self.run_folder + '/topology.png'
            plot_matrix(connection_matrix, topology_target_file)

        # Set paths for checkpoints, metrics, and scheduler
        self.checkpoint_folder = self.run_folder + '/checkpoints'
        self.metrics_folder = self.run_folder + '/metrics.pkl'
        self.scheduler_file = self.run_folder + '/scheduler.pkl'

        if resume_run:
            # Load metrics if resuming a run
            with open(self.metrics_folder, 'rb') as f:
                self.metrics = pickle.load(f)
        else:
            # Initialize metrics
            self.metrics = {
                'epsilon_history': [1.0],
                'delay_without_drop_rewards_history': [],
                'energy_rewards_history': [],
                'rewards_history': [],
                'task_drop_ratio_history': [],
                'actions_history': []
            }
            with open(self.metrics_folder, 'wb') as f:
                pickle.dump(self.metrics, f)

            # Initialize scheduler
            scheduler_choices = {
                'constant': constant,
                'Linear': Linear(start=hyperparameters['learning_rate'],
                                 end=hyperparameters['learning_rate_end'],
                                 number_of_epochs=hyperparameters['lr_scheduler_epochs'])
            }
            scheduler = scheduler_choices[hyperparameters['scheduler_choice']]
            with open(self.scheduler_file, 'wb') as f:
                pickle.dump(scheduler, f)

        # Initialize reward and task lists
        self.delay_without_drop_rewards = []
        self.energy_rewards = []
        self.rewards = []
        self.tasks_dropped = []
        self.tasks_arrived = []

        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def get_epsilon(self):
        return self.metrics['epsilon_history'][-1]

    def get_checkpoint_folder(self):
        return self.checkpoint_folder
    
    def get_scheduler_file(self):  
        return self.scheduler_file
    
    def get_hyperparameters(self):
        with open(self.hyperparameters_file) as f:
            hyperparameters = json.load(f)
        hyperparameters['epsilon'] = self.get_epsilon()
        return hyperparameters

    def store_step(self, info):
        # Store step information
        self.delay_without_drop_rewards.append(info['delay_without_drop_rewards'])
        self.energy_rewards.append(info['energy_rewards'])
        self.rewards.append(info['rewards'])
        self.tasks_dropped.append(info['tasks_dropped'])
        self.tasks_arrived.append(info['tasks_arrived'])
        
    def store_episode(self, epsilon, actions):
        # Store episode information
        episode_delay_without_drop_rewards = np.vstack(self.delay_without_drop_rewards)
        episode_delay_without_drop_rewards = np.sum(episode_delay_without_drop_rewards, axis=0)
        self.metrics['delay_without_drop_rewards_history'].append(episode_delay_without_drop_rewards)
        
        episode_energy_rewards = np.vstack(self.energy_rewards)
        episode_energy_rewards = np.sum(episode_energy_rewards, axis=0)
        self.metrics['energy_rewards_history'].append(episode_energy_rewards)
        
        episode_rewards = np.vstack(self.rewards)
        episode_rewards = np.sum(episode_rewards, axis=0)
        self.metrics['rewards_history'].append(episode_rewards)
        
        episode_tasks_arrived = np.vstack(self.tasks_arrived)
        episode_tasks_arrived = np.sum(episode_tasks_arrived, axis=0)
        episode_tasks_drop = np.vstack(self.tasks_dropped)
        episode_tasks_drop = np.sum(episode_tasks_drop, axis=0)
        episode_task_drop_ratio = episode_tasks_drop / episode_tasks_arrived
        self.metrics['task_drop_ratio_history'].append(episode_task_drop_ratio)
        
        self.metrics['actions_history'].append(actions)
        
        epochs = len(self.metrics['rewards_history'])
        self.metrics['epsilon_history'].append(epsilon)
        
        with open(self.metrics_folder, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Reset lists for the next episode
        self.delay_without_drop_rewards = []
        self.energy_rewards = []
        self.rewards = []
        self.tasks_arrived = []
        self.tasks_dropped = []
        
        score, average_score = np.mean(self.metrics['rewards_history'][-1]), np.mean(self.metrics['rewards_history'][-self.average_window:])
        print(f'Epoch: {epochs} \tScore: {score:.3f} \tAverage Score: {average_score:.3f} \tEpsilon: {epsilon:.3f}')
        
        return average_score

    def plot_and_save(self, key):
        if key not in self.metrics:
            print(f"No agent_actions found for key '{key}'")
            return
        list_of_arrays = self.metrics[key]
        stacked_arrays = np.vstack(list_of_arrays)
        transposed_arrays = stacked_arrays.T
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(transposed_arrays):
            plt.plot(column, label=f'agent {i} {key} ', linestyle='--')
        mean_values = np.mean(transposed_arrays, axis=0)
        plt.plot(mean_values, label='Mean', color='red', linewidth=6)
        plt.legend()
        plt.title(f'Plot of {key} and Their Mean')
        plt.savefig(f'{self.run_folder}/{key}.png')
        plt.close()

    def moving_average(self, a):
        return [np.mean(a[max(0, i - self.average_window):i]) for i in range(1, len(a))]

    def plot_and_save_moving_avg(self, key):
        if key not in self.metrics:
            print(f"No agent_actions found for key '{key}'")
            return
        list_of_arrays = self.metrics[key]
        stacked_arrays = np.vstack(list_of_arrays)
        transposed_arrays = stacked_arrays.T

        plt.figure(figsize=(10, 6))
        means = []
        for i, column in enumerate(transposed_arrays):
            moving_avg = self.moving_average(column)
            plt.plot(moving_avg, label=f'agent {i}', linestyle='--')
            means.append(moving_avg)
        means = np.mean(means, axis=0)
        plt.plot(means, label='Mean', color='red', linewidth=6)
        plt.legend()
        plt.title(f'Plot of Moving Average of {key}')
        plt.savefig(f'{self.run_folder}/{key}_moving_average.png')
        plt.close()
        
    def plot_actions(self):
        def plot_single_action(agent_actions, title, savefile):
            local_values = [d['local'] for d in agent_actions]
            horizontal_values = [d['horisontal'] for d in agent_actions]  # Note the typo in 'horizontal'
            cloud_values = [d['cloud'] for d in agent_actions]

            time = list(range(len(agent_actions)))

            plt.figure(figsize=(10, 6))
            plt.plot(time, local_values, label='Local', marker='o')
            plt.plot(time, horizontal_values, label='Horizontal', marker='o')
            plt.plot(time, cloud_values, label='Cloud', marker='o')

            plt.title(title)
            plt.xlabel('Episode')
            plt.ylabel('Number Of Time chosen')
            plt.legend()
            plt.savefig(savefile)
            plt.close()

        actions_folder = f'{self.run_folder}/actions'
        os.makedirs(actions_folder, exist_ok=True)
        for agent in range(len(self.metrics['actions_history'][0])):
            agent_actions = [row[agent] for row in self.metrics['actions_history']]
            title = f'Actions of agent {agent}'
            savefile = f'{actions_folder}/actions_{agent}.png'
            plot_single_action(agent_actions, title, savefile)
        total_actions = sum_dicts_in_positions(self.metrics['actions_history'])
        title = 'Total Actions'
        savefile = f'{actions_folder}/actions_total.png'
        plot_single_action(total_actions, title, savefile)
          
        return
        
    def plot_metrics(self):
        for key in self.metrics.keys():
            if key in self.plotable_metrics:
                self.plot_and_save(key)
                self.plot_and_save_moving_avg(key)
        self.plot_actions()

    def get_run_folder(self):
        return self.run_folder
     
    def get_rewards_history(self):
        return self.metrics['rewards_history']
