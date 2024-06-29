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
    def __init__(self,
                 log_folder='log_folder',
                 hyperparameters_source_file=None,
                 resume_run=None,
                 average_window=500):
        
        self.plotable_metrics = ['rewards_history','task_drop_ratio_history']
        self.log_folder = log_folder
        self.average_window = average_window
        os.makedirs(log_folder, exist_ok=True)

        if resume_run :
            self.run_folder = os.path.join(log_folder,resume_run)
            self.hyperparameters_file = os.path.join(self.run_folder,'hyperparameters.json')
            

        else :
            with open(hyperparameters_source_file) as f:
                hyperparameters = json.load(f)
            index_filepath  =log_folder+'/index.txt'
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
            self.run_folder = log_folder+'/run_'+str(run_index)
            os.makedirs(self.run_folder,exist_ok=True)
            self.hyperparameters_file = self.run_folder+'/hyperparameters.json'
            json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)
            with open(self.hyperparameters_file, "w") as outfile:
                    outfile.write(json_object)
                    
            connection_matrix = hyperparameters['connection_matrix']
            topology_target_file = self.run_folder + '/topology.png'

            plot_matrix(connection_matrix,topology_target_file)
            
            
        
        
                
        self.checkpoint_folder = self.run_folder+'/checkpoints'
        self.metrics_folder = self.run_folder+'/metrics.pkl'
        
        self.scheduler_file = self.run_folder+'/scheduler.pkl'

        if resume_run:
            with open(self.metrics_folder, 'rb') as f:
                self.metrics = pickle.load(f)
                
        else:
                        
            self.metrics ={}
            self.metrics['epsilon_history'] =[1.0]
            self.metrics['rewards_history'] =[]
            self.metrics['task_drop_ratio_history'] =[]
            self.metrics['actions_history'] = []


            with open(self.metrics_folder, 'wb') as f:
                pickle.dump(self.metrics, f)

            scheduler_choices ={
                'constant': constant,
                'Linear': Linear(start=hyperparameters['learning_rate'],
                                end=hyperparameters['learning_rate_end'],
                                    number_of_epochs=hyperparameters['lr_scheduler_epochs'])
                }
            
            scheduler =scheduler_choices[hyperparameters['scheduler_choice']]
            
            
            with open(self.scheduler_file, 'wb') as f:
                pickle.dump(scheduler, f)
                
                

        self.rewards = []
        self.tasks_dropped =[]
        self.tasks_arrived =[]

        os.makedirs(self.checkpoint_folder,exist_ok=True)
        
        
       

        
        
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
    

    def store_step(self,info):
        self.rewards.append(info['rewards'])
        
        self.tasks_dropped.append(info['tasks_dropped'])
        self.tasks_arrived.append(info['tasks_arrived'])
        
    def store_episode(self,epsilon,actions):
        episode_rewards = np.vstack(self.rewards)
        episode_rewards=  np.sum(episode_rewards,axis=0)
        self.metrics['rewards_history'].append(episode_rewards)
        
        
        episode_tasks_arrived = np.vstack(self.tasks_arrived)
        episode_tasks_arrived = np.sum(episode_tasks_arrived,axis=0)
        episode_tasks_drop = np.vstack(self.tasks_dropped)
        episode_tasks_drop = np.sum(episode_tasks_drop,axis=0)
        episode_task_drop_ratio = episode_tasks_drop/episode_tasks_arrived
        self.metrics['task_drop_ratio_history'].append(episode_task_drop_ratio)
        
        
        self.metrics['actions_history'].append(actions)
        
        epochs = len(self.metrics['rewards_history'])
        self.metrics['epsilon_history'].append(epsilon)
        
        with open(self.metrics_folder, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        self.rewards =[]
        self.tasks_arrived =[]
        self.tasks_dropped =[]
        
        score, average_score = np.mean(self.metrics['rewards_history'][-1]), np.mean(self.metrics['rewards_history'][-self.average_window:])
        print(f'Epoch: {epochs} \tScore: {score:.3f} \tAverage Score: {average_score:.3f} \tEpsilon: {epsilon:.3f}')
        

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
        return  [np.mean(a[max(0,i-self.average_window):i]) for i in range(1,len(a))]

        
    def plot_and_save_moving_avg(self, key):
        if key not in self.metrics:
            print(f"No agent_actions found for key '{key}'")
            return
      

        list_of_arrays = self.metrics[key]

        stacked_arrays = np.vstack(list_of_arrays)

        transposed_arrays = stacked_arrays.T

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot the moving average of each column
        means = []  # List to store the means of the moving averages
        for i, column in enumerate(transposed_arrays):
            moving_avg = self.moving_average(column)  # Change n to your desired window size
            plt.plot(moving_avg, label=f'agent {i}', linestyle='--')
            means.append(moving_avg)
        means = np.mean(means, axis=0)
        # Plot the mean of the moving averages
        plt.plot(means, label='Mean', color='red',linewidth=6)

        # Add a legend and title
        plt.legend()
        plt.title(f'Plot of Moving Average of {key}')

        # Save the plot as a PNG file
        plt.savefig(f'{self.run_folder}/{key}_moving_average.png')
        plt.close()
        
    def plot_actions(self):
        def plot_single_action(agent_actions,title,savefile):
            local_values = [d['local'] for d in agent_actions]
            horizontal_values = [d['horisontal'] for d in agent_actions]  # Note the typo in 'horizontal'
            cloud_values = [d['cloud'] for d in agent_actions]

         # Time points
            time = list(range(len(agent_actions)))

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(time, local_values, label='Local', marker='o')
            plt.plot(time, horizontal_values, label='Horizontal', marker='o')
            plt.plot(time, cloud_values, label='Cloud', marker='o')

            # Adding plot decorations
            plt.title(title)
            plt.xlabel('Episode')
            plt.ylabel('Number Of Time chosen')
            plt.legend()

            plt.savefig(savefile)
            plt.close()
        actions_folder = f'{self.run_folder}/actions'
        os.makedirs(actions_folder,exist_ok=True)
        for agent in range(len(self.metrics['actions_history'][0])):
            agent_actions=  [row[agent] for row in self.metrics['actions_history']]
            title = 'Actions of agent {agent}'
            savefile = f'{actions_folder}/actions_{agent}.png'
            plot_single_action(agent_actions,title,savefile)
        total_actions = sum_dicts_in_positions(self.metrics['actions_history'])
        title = 'Total Actions'
        savefile = f'{actions_folder}/actions_total.png'
        plot_single_action(total_actions,title,savefile)
          
        return
        
    def plot_metrics(self):
        for key in self.metrics.keys():
            if key in self.plotable_metrics:
                self.plot_and_save(key)
                self.plot_and_save_moving_avg(key)
        self.plot_actions()

        