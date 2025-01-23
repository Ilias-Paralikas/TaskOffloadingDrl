import argparse
import json
import  os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topology_generators import SkipConnections, FullyConnected

NUMBER_OF_CLOUDS =1

def comma_seperated_string_to_list(comma_seperated_String,dtype):
        if comma_seperated_String is None:
                return []
        return [dtype(x) for x in comma_seperated_String.split(',')]

def fill_array(string, length, default_value,dtype):
        array  = comma_seperated_string_to_list(string,dtype)
        values=  [default_value for _ in range(length)]
        for i in range(len(array)):
                values[i] = array[i]
        return values

def main():
        
        '''
        
        This is a description of the command line argument pattern used in this script.
        When the script is run, it processes command line arguments to generate a hyperparameters JSON file.
        The arguments follow a pattern where there are both default and specific values for many parameters,
        allowing for flexible configuration of the network servers.
        
        Notes 1:
        
        In this code, many parameters have both a "default" version 
        and a specific version (without "default" in the name) to provide flexibility in configuration. 

        Take these two parameters as an example:
                parser.add_argument('--default_private_cpu_capacity', type=float, default=5)
                parser.add_argument('--private_cpu_capacities', type=str, default=None)
        This pattern exists because:

        Default Value: The default_private_cpu_capacity (=5) applies to all servers if no specific values are provided
        Per-Server Configuration: private_cpu_capacities allows setting different values for each 
        server using a comma-separated string (e.g., "3,4,5,6")
        
        If you don't specify private_cpu_capacities, all servers get the default value (5)
        If you specify private_cpu_capacities="3,4,5":
        First 3 servers get values 3, 4, and 5 respectively
        Any remaining servers get the default value (5)
        This design allows both:
        Simple configuration (just set the default)
        Fine-grained control (specify individual values per server)
        
        
        
        Notes 2:
        
        Some parameters have min, max, and distribution settings because they represent randomly 
        generated values that need to be sampled according to specific probability distributions. Here's why:


        Parameters like task_size, timeout_delay, priority need to vary randomly during simulation
        Each task gets random values within the specified ranges

        min: Lower bound of the random value
        max: Upper bound of the random value
        distribution: How to sample between min and max (e.g., 'uniform', 'constant')
        Here's an example of how these parameters are defined in the script:
                # Task size configuration
                parser.add_argument('--default_task_size_mins', type=int, default=2)
                parser.add_argument('--task_size_maxs', type=str, default=None)
                parser.add_argument('--task_size_distributions', type=str, default='uniform')
        Common Distributions:
        uniform: Random values evenly distributed between min and max
        constant: Always uses the same value (min=max)
        This allows simulating realistic scenarios where task properties vary randomly within configured bounds.
        '''
        
        parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
        parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='Output path for the generated hyperparameters JSON file')
        
        parser.add_argument('--number_of_servers', type=int, default=20, help='Total number of edge servers in the network')

        parser.add_argument('--default_private_cpu_capacity', type=float, default=5, help='Default CPU capacity for private processing')
        parser.add_argument('--private_cpu_capacities', type=str, default=None, help='Comma-separated list of CPU capacities for private processing per server')
        
        parser.add_argument('--default_public_cpu_capacity', type=float, default=5, help='Default CPU capacity for public processing')
        parser.add_argument('--public_cpu_capacities', type=str, default=None, help='Comma-separated list of CPU capacities for public processing per server')
        
        parser.add_argument('--episode_time', type=int, default=100, help='Duration of each training episode')
        parser.add_argument('--time_step', type=int, default=0.1, help='Time step for simulation discretization')
        
        parser.add_argument('--static_frequency', type=int, default=0, help='Frequency of static decision making (0 for dynamic)')
        
        parser.add_argument('--cloud_computational_capacity', type=float, default=30, help='Computational capacity of the cloud server')
        
   
        parser.add_argument('--default_private_queue_waiting_time_consumptions', type=float, default=0.1, help='Default energy consumption rate while waiting in private queue')
        parser.add_argument('--private_queue_waiting_time_consumptions', type=str, default=None, help='Comma-separated list of energy consumption rates while waiting in private queue per server')

        parser.add_argument('--default_private_queue_step_consumptions', type=float, default=1, help='Default energy consumption per step in private queue')
        parser.add_argument('--private_queue_step_consumptions', type=str, default=None, help='Comma-separated list of energy consumption per step in private queue per server')
        
        parser.add_argument('--default_public_queue_waiting_time_consumptions', type=float, default=0.2, help='Default energy consumption rate while waiting in public queue')
        parser.add_argument('--public_queue_waiting_time_consumptions', type=str, default=None, help='Comma-separated list of energy consumption rates while waiting in public queue per server')
        
        parser.add_argument('--default_public_queue_step_consumptions', type=float, default=2, help='Default energy consumption per step in public queue')
        parser.add_argument('--public_queue_step_consumptions', type=str, default=None, help='Comma-separated list of energy consumption per step in public queue per server')
        
        parser.add_argument('--default_offloading_queue_waiting_time_consumptions', type=float, default=0.3, help='Default energy consumption rate while waiting in offloading queue')
        parser.add_argument('--offloading_queue_waiting_time_consumptions', type=str, default=None, help='Comma-separated list of energy consumption rates while waiting in offloading queue per server')
        
        parser.add_argument('--default_offloading_queue_step_consumptions', type=float, default=3, help='Default energy consumption per step in offloading queue')
        parser.add_argument('--offloading_queue_step_consumptions', type=str, default=None, help='Comma-separated list of energy consumption per step in offloading queue per server')
        
        parser.add_argument('--cloud_waiting_time_consumption', type=float, default=0.4, help='Energy consumption rate while waiting in cloud queue')
        parser.add_argument('--cloud_step_consumption', type=float, default=4, help='Energy consumption per step in cloud queue')
        
        parser.add_argument('--delay_to_energy_ratio', type=float, default=0.5, help='Weight ratio between delay and energy in the cost function')
       
        parser.add_argument('--default_task_arrive_probabilities', type=float, default=0.9, help='Default probability of task arrival per time step')
        parser.add_argument('--task_arrive_probabilities', type=str, default=None, help='Comma-separated list of task arrival probabilities per server')
        
        parser.add_argument('--default_task_size_mins', type=int, default=2, help='Default minimum task size')
        parser.add_argument('--task_size_mins', type=str, default=None, help='Comma-separated list of minimum task sizes per server')
        parser.add_argument('--default_task_size_maxs', type=int, default=5, help='Default maximum task size')
        parser.add_argument('--task_size_maxs', type=str, default=None, help='Comma-separated list of maximum task sizes per server')
        parser.add_argument('--task_size_distributions', type=str, default='uniform', help='Distribution type for task sizes (uniform, constant, etc)')
        
        parser.add_argument('--default_timeout_delay_mins', type=int, default=10, help='Default minimum timeout delay')
        parser.add_argument('--timeout_delay_mins', type=str, default=None, help='Comma-separated list of minimum timeout delays per server')
        parser.add_argument('--default_timeout_delay_maxs', type=int, default=10, help='Default maximum timeout delay')
        parser.add_argument('--timeout_delay_maxs', type=str, default=None, help='Comma-separated list of maximum timeout delays per server')
        parser.add_argument('--timeout_delay_distributions', type=str, default='constant', help='Distribution type for timeout delays')
        
        parser.add_argument('--default_priotiry_mins', type=int, default=1, help='Default minimum task priority')
        parser.add_argument('--priotiry_mins', type=str, default=None, help='Comma-separated list of minimum task priorities per server')
        parser.add_argument('--default_priotiry_maxs', type=int, default=1, help='Default maximum task priority')
        parser.add_argument('--priotiry_maxs', type=str, default=None, help='Comma-separated list of maximum task priorities per server')
        parser.add_argument('--priotiry_distributions', type=str, default='constant', help='Distribution type for task priorities')
        
        parser.add_argument('--default_computational_density_mins', type=int, default=0.297, help='Default minimum computational density')
        parser.add_argument('--computational_density_mins', type=str, default=None, help='Comma-separated list of minimum computational densities per server')
        parser.add_argument('--default_computational_density_maxs', type=int, default=0.297, help='Default maximum computational density')
        parser.add_argument('--computational_density_maxs', type=str, default=None, help='Comma-separated list of maximum computational densities per server')
        parser.add_argument('--computational_density_distributions', type=str, default='constant', help='Distribution type for computational densities')
        
        parser.add_argument('--default_drop_penalty_mins', type=int, default=40, help='Default minimum penalty for dropping tasks')
        parser.add_argument('--drop_penalty_mins', type=str, default=None, help='Comma-separated list of minimum drop penalties per server')
        parser.add_argument('--default_drop_penalty_maxs', type=int, default=40, help='Default maximum penalty for dropping tasks')
        parser.add_argument('--drop_penalty_maxs', type=str, default=None, help='Comma-separated list of maximum drop penalties per server')
        parser.add_argument('--drop_penalty_distributions', type=str, default='constant', help='Distribution type for drop penalties')
         
        parser.add_argument('--horizontal_capacities_min', type=float, default=10, help='Minimum capacity for horizontal connections between servers')
        parser.add_argument('--horizontal_capacities_max', type=float, default=10, help='Maximum capacity for horizontal connections between servers')
        parser.add_argument('--horizontal_capacities_distribution', type=str, default='constant', help='Distribution type for horizontal connection capacities')
        
        parser.add_argument('--cloud_capacities_min', type=float, default=20, help='Minimum capacity for cloud connections')
        parser.add_argument('--cloud_capacities_max', type=float, default=30, help='Maximum capacity for cloud connections')
        parser.add_argument('--cloud_capacities_distribution', type=str, default='constant', help='Distribution type for cloud connection capacities')
        
        parser.add_argument('--skip_connections', type=int, default=5, help='Number of skip connections in the network topology')
        
        parser.add_argument('--topology_generator', type=str, default='skip_connections', help='Type of topology generator (skip_connections or fully_connected)')
        parser.add_argument('--symetric', type=bool, default=False, help='Whether the network topology should be symmetric')
        
        parser.add_argument('--decision_makers', type=str, default='rule_based', help='Type of decision making algorithm')
        parser.add_argument('--hidden_layers', type=str, default='1024,1024,1024', help='Comma-separated list of hidden layer sizes for neural network')
        parser.add_argument('--lstm_layers', type=int, default=20, help='Number of LSTM layers')
        parser.add_argument('--lstm_time_step', type=int, default=10, help='Time steps for LSTM memory')
        parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for neural network')
        parser.add_argument('--dueling', type=bool, default=True, help='Whether to use dueling architecture in DQN')
        parser.add_argument('--epsilon_decrement', type=float, default=100, help='Decrement rate for epsilon in epsilon-greedy exploration')
        parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final value for epsilon in epsilon-greedy exploration')
        parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
        parser.add_argument('--learning_rate', type=float, default=1e-6, help='Initial learning rate')
        parser.add_argument('--learning_rate_end', type=float, default=1e-7, help='Final learning rate')
        parser.add_argument('--scheduler_choice', type=str, default='constant', help='Learning rate scheduler type')
        parser.add_argument('--lr_scheduler_epochs', type=int, default=2000, help='Number of epochs for learning rate scheduling')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimization algorithm choice')
        parser.add_argument('--loss_function', type=str, default='MSELoss', help='Loss function for training')
        parser.add_argument('--save_model_frequency', type=int, default=10, help='Frequency of model checkpointing')
        parser.add_argument('--update_weight_percentage', type=float, default=1.0, help='Percentage of weights to update in each training step')
        parser.add_argument('--memory_size', type=int, default=10000, help='Size of replay memory buffer')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--replace_target_iter', type=int, default=50, help='Frequency of target network updates')
        
        parser.add_argument('--championship_windows', type=str, default='1,2,5,10,20,50,100,200,500', help='Comma-separated list of window sizes for championship evaluation')
        parser.add_argument('--championship_start', type=int, default=1, help='Episode number to start championship evaluation')
        args = parser.parse_args()
        
        
        private_cpu_capacities = fill_array(args.private_cpu_capacities,args.number_of_servers,args.default_private_cpu_capacity,float)
        public_cpu_capacities = fill_array(args.public_cpu_capacities,args.number_of_servers,args.default_public_cpu_capacity,float)
        cloud_computational_capacity = args.cloud_computational_capacity *args.time_step
        
        private_queue_waiting_time_consumptions = fill_array(args.private_queue_waiting_time_consumptions,args.number_of_servers,args.default_private_queue_waiting_time_consumptions,float)
        private_queue_step_consumptions = fill_array(args.private_queue_step_consumptions,args.number_of_servers,args.default_private_queue_step_consumptions,float)
        public_queue_waiting_time_consumptions = fill_array(args.public_queue_waiting_time_consumptions,args.number_of_servers,args.default_public_queue_waiting_time_consumptions,float)
        public_queue_step_consumptions = fill_array(args.public_queue_step_consumptions,args.number_of_servers,args.default_public_queue_step_consumptions,float)
        offloading_queue_waiting_time_consumptions = fill_array(args.offloading_queue_waiting_time_consumptions,args.number_of_servers,args.default_offloading_queue_waiting_time_consumptions,float)
        offloading_queue_step_consumptions = fill_array(args.offloading_queue_step_consumptions,args.number_of_servers,args.default_offloading_queue_step_consumptions,float)
        cloud_waiting_time_consumption = args.cloud_waiting_time_consumption*args.time_step
        cloud_step_consumption = args.cloud_step_consumption*args.time_step
        delay_to_energy_ratio= args.delay_to_energy_ratio
      
        task_arrive_probabilities = fill_array(args.task_arrive_probabilities,args.number_of_servers,args.default_task_arrive_probabilities,float)
        task_size_mins = fill_array(args.task_size_mins,args.number_of_servers,args.default_task_size_mins,int)
        task_size_maxs = fill_array(args.task_size_maxs,args.number_of_servers,args.default_task_size_maxs,int)
        task_size_distributions = fill_array(args.task_size_distributions,args.number_of_servers,args.task_size_distributions,str)
        timeout_delay_mins = fill_array(args.timeout_delay_mins,args.number_of_servers,args.default_timeout_delay_mins,int)
        timeout_delay_maxs = fill_array(args.timeout_delay_maxs,args.number_of_servers,args.default_timeout_delay_maxs,int)
        timeout_delay_distributions = fill_array(args.timeout_delay_distributions,args.number_of_servers,args.timeout_delay_distributions,str)
        priotiry_mins = fill_array(args.priotiry_mins,args.number_of_servers,args.default_priotiry_mins,int)
        priotiry_maxs = fill_array(args.priotiry_maxs,args.number_of_servers,args.default_priotiry_maxs,int)
        priotiry_distributions = fill_array(args.priotiry_distributions,args.number_of_servers,args.priotiry_distributions,str)
        computational_density_mins = fill_array(args.computational_density_mins,args.number_of_servers,args.default_computational_density_mins,int)
        computational_density_maxs = fill_array(args.computational_density_maxs,args.number_of_servers,args.default_computational_density_maxs,int)
        computational_density_distributions = fill_array(args.computational_density_distributions,args.number_of_servers,args.computational_density_distributions,str)
        drop_penalty_mins = fill_array(args.drop_penalty_mins,args.number_of_servers,args.default_drop_penalty_mins,int)
        drop_penalty_maxs = fill_array(args.drop_penalty_maxs,args.number_of_servers,args.default_drop_penalty_maxs,int)
        drop_penalty_distributions = fill_array(args.drop_penalty_distributions,args.number_of_servers,args.drop_penalty_distributions,str)
        

        
        
        
        topology_generator_choices = {
                'skip_connections':SkipConnections,
                'fully_connected':FullyConnected
        }
        
        
        hidden_layers = comma_seperated_string_to_list(args.hidden_layers,int)
        

        topology_generator = topology_generator_choices[args.topology_generator](
                number_of_servers=args.number_of_servers,
                horizontal_capacities_min=args.horizontal_capacities_min*args.time_step,
                horizontal_capacities_max=args.horizontal_capacities_max*args.time_step,
                horizontal_capacities_distribution=args.horizontal_capacities_distribution,
                cloud_capacities_min=args.cloud_capacities_min*args.time_step,
                cloud_capacities_max=args.cloud_capacities_max*args.time_step,
                cloud_capacities_distribution=args.cloud_capacities_distribution,
                skip_connections=args.skip_connections,
                symetric=args.symetric
        )
        connection_matrix = topology_generator.create_topology()
        connection_matrix = connection_matrix.tolist()
        
        mull_array = lambda  arr,x : [x *e for e in arr]

        hyperparameters = { 
                "number_of_servers":args.number_of_servers,
                "private_cpu_capacities":mull_array(private_cpu_capacities,args.time_step),
                "public_cpu_capacities":mull_array(public_cpu_capacities,args.time_step),
                "episode_time":args.episode_time,
                "static_frequency":args.static_frequency,
                "cloud_computational_capacity":cloud_computational_capacity,
                "private_queue_waiting_time_consumptions":mull_array(private_queue_waiting_time_consumptions,args.time_step),
                "private_queue_step_consumptions":mull_array(private_queue_step_consumptions,args.time_step),
                "public_queue_waiting_time_consumptions":mull_array(public_queue_waiting_time_consumptions,args.time_step),
                "public_queue_step_consumptions":mull_array(public_queue_step_consumptions,args.time_step),
                "offloading_queue_waiting_time_consumptions":mull_array(offloading_queue_waiting_time_consumptions,args.time_step),
                "offloading_queue_step_consumptions":mull_array(offloading_queue_step_consumptions,args.time_step),
                "cloud_waiting_time_consumption":cloud_waiting_time_consumption,
                "cloud_step_consumption":cloud_step_consumption,
                "delay_to_energy_ratio":delay_to_energy_ratio,
                "task_arrive_probabilities":task_arrive_probabilities,
                "task_size_mins":task_size_mins,
                "task_size_maxs":task_size_maxs,
                "task_size_distributions":task_size_distributions,
                "timeout_delay_mins":timeout_delay_mins,
                "timeout_delay_maxs":timeout_delay_maxs,
                "timeout_delay_distributions":timeout_delay_distributions,
                "priotiry_mins":priotiry_mins,
                "priotiry_maxs":priotiry_maxs,
                "priotiry_distributions":priotiry_distributions,
                "computational_density_mins":computational_density_mins,
                "computational_density_maxs":computational_density_maxs,
                "computational_density_distributions":computational_density_distributions,
                "drop_penalty_mins":drop_penalty_mins,
                "drop_penalty_maxs":drop_penalty_maxs,
                "drop_penalty_distributions":drop_penalty_distributions,
                "connection_matrix" :connection_matrix,
                "decision_makers":args.decision_makers,
                "hidden_layers":hidden_layers,
                "lstm_layers":args.lstm_layers,
                "lstm_time_step":args.lstm_time_step,
                "dropout_rate":args.dropout_rate,
                "dueling":args.dueling,
                "epsilon_decrement":args.epsilon_decrement,
                "epsilon_end":args.epsilon_end,
                "gamma":args.gamma,
                "learning_rate":args.learning_rate,
                "learning_rate_end":args.learning_rate_end,
                "scheduler_choice":args.scheduler_choice,
                "lr_scheduler_epochs":args.lr_scheduler_epochs,
                "optimizer":args.optimizer,
                "loss_function":args.loss_function,
                "save_model_frequency":args.save_model_frequency,
                "update_weight_percentage":args.update_weight_percentage,
                "memory_size":int(args.memory_size),
                "batch_size":args.batch_size  ,
                "replace_target_iter":args.replace_target_iter,
                "championship_windows":comma_seperated_string_to_list(args.championship_windows,int),
                "championship_start":args.championship_start
        }
        
        json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)

        
        with open(args.hyperparameters_file, 'w') as outfile:
            outfile.write(json_object)
if __name__ =="__main__":
    main()