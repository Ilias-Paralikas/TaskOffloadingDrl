import os 
import json
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_generators import plot_matrix
class BookKeeper:
    def __init__(self,
                 log_folder='log_folder',
                 hyperparameters_source_file=None,
                 resume_run=None,
                 average_window=500):
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
        
        if resume_run:
            with open(self.metrics_folder, 'rb') as f:
                self.metrics = pickle.load(f)
        else:
                        
            self.metrics ={}
            self.metrics['epsilon_history'] =[1.0]
            self.metrics['rewards_history'] =[]

                
            with open(self.metrics_folder, 'wb') as f:
                pickle.dump(self.metrics, f)
                

        self.rewards = []

        os.makedirs(self.checkpoint_folder,exist_ok=True)
        
        
        
    def get_epsilon(self):
        return self.metrics['epsilon_history'][-1]

    def get_checkpoint_folder(self):
        return self.checkpoint_folder
    
    def get_hyperparameters(self):
        with open(self.hyperparameters_file) as f:
            hyperparameters = json.load(f)
        hyperparameters['epsilon'] = self.get_epsilon()
        return hyperparameters
    

    def store_step(self,info):
        self.rewards.append(info['rewards'])
        
    def store_episode(self,epsilon):
        episode_rewards = np.vstack(self.rewards)
        episode_rewards=  np.sum(episode_rewards,axis=0)
        self.metrics['rewards_history'].append(episode_rewards)
        
        epochs = len(self.metrics['rewards_history'])
        self.metrics['epsilon_history'].append(epsilon)
        
        with open(self.metrics_folder, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        self.rewards =[]
        
        score, average_score = np.mean(self.metrics['rewards_history'][-1]), np.mean(self.metrics['rewards_history'][-self.average_window:])
        print(f'Epoch: {epochs} Score: {score:.3f} Average Score: {average_score:.3f} Epsilon: {epsilon:.3f}')
        

    def plot_and_save(self, key):
        if key not in self.metrics:
            print(f"No data found for key '{key}'")
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
            print(f"No data found for key '{key}'")
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
        
        
    def plot_metrics(self):
        for key in self.metrics.keys():
            self.plot_and_save(key)
            self.plot_and_save_moving_avg(key)

        