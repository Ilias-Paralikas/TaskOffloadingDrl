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
        parser = argparse.ArgumentParser(description='Script Configuration via Command Line')
        # file to get the hyperparameters from
        parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='Path to the hyperparameters file')
        
        parser.add_argument('--number_of_servers', type=int, default=20, help='Number of servers in the system')

        parser.add_argument('--default_private_cpu_capacity', type=float, default=5, help='Number of servers in the system')
        parser.add_argument('--private_cpu_capacities', type=str, default=None, help='Number of servers in the system')
        
        parser.add_argument('--default_public_cpu_capacity', type=float, default=5, help='Number of servers in the system')
        parser.add_argument('--public_cpu_capacities', type=str, default=None, help='Number of servers in the system')
        
        parser.add_argument('--episode_time', type=int, default=100, help='Number of servers in the system')
        parser.add_argument('--time_step', type=int, default=0.1, help='Number of servers in the system')
        
        parser.add_argument('--static_frequency', type=int, default=0, help='Number of servers in the system')
        
        parser.add_argument('--cloud_computational_capacity', type=float, default=30, help='Number of servers in the system')
        
        parser.add_argument('--default_task_arrive_probabilities', type=float, default=0.5, help='Number of servers in the system')
        parser.add_argument('--task_arrive_probabilities', type=str, default=None, help='Number of servers in the system')
        
        parser.add_argument('--default_task_size_mins', type=int, default=2, help='Number of servers in the system')
        parser.add_argument('--task_size_mins', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--default_task_size_maxs', type=int, default=5, help='Number of servers in the system')
        parser.add_argument('--task_size_maxs', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--task_size_distributions', type=str, default='uniform', help='Number of servers in the system')
        
        parser.add_argument('--default_timeout_delay_mins', type=int, default=10, help='Number of servers in the system')
        parser.add_argument('--timeout_delay_mins', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--default_timeout_delay_maxs', type=int, default=10, help='Number of servers in the system')
        parser.add_argument('--timeout_delay_maxs', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--timeout_delay_distributions', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--default_priotiry_mins', type=int, default=1, help='Number of servers in the system')
        parser.add_argument('--priotiry_mins', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--default_priotiry_maxs', type=int, default=1, help='Number of servers in the system')
        parser.add_argument('--priotiry_maxs', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--priotiry_distributions', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--default_computational_density_mins', type=int, default=0.297, help='Number of servers in the system')
        parser.add_argument('--computational_density_mins', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--default_computational_density_maxs', type=int, default=0.297, help='Number of servers in the system')
        parser.add_argument('--computational_density_maxs', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--computational_density_distributions', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--default_drop_penalty_mins', type=int, default=40, help='Number of servers in the system')
        parser.add_argument('--drop_penalty_mins', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--default_drop_penalty_maxs', type=int, default=40, help='Number of servers in the system')
        parser.add_argument('--drop_penalty_maxs', type=str, default=None, help='Number of servers in the system')
        parser.add_argument('--drop_penalty_distributions', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--horizontal_capacities_min', type=float, default=10, help='Number of servers in the system')
        parser.add_argument('--horizontal_capacities_max', type=float, default=10, help='Number of servers in the system')
        parser.add_argument('--horizontal_capacities_distribution', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--cloud_capacities_min', type=float, default=20, help='Number of servers in the system')
        parser.add_argument('--cloud_capacities_max', type=float, default=30, help='Number of servers in the system')
        parser.add_argument('--cloud_capacities_distribution', type=str, default='constant', help='Number of servers in the system')
        
        parser.add_argument('--skip_connections', type=int, default=3, help='Number of servers in the system')
        
        parser.add_argument('--topology_generator', type=str, default='skip_connections', help='Number of servers in the system')
        parser.add_argument('--symetric', type=bool, default=True, help='Number of servers in the system')
        
        parser.add_argument('--decision_makers', type=str, default='drl', help='Number of servers in the system')
        parser.add_argument('--hidden_layers', type=str, default='1024,1024,1024', help='comma-separated integers')
        parser.add_argument('--lstm_layers', type=int, default=20, help='Integer')
        parser.add_argument('--lstm_time_step', type=int, default=10, help='Integer')
        parser.add_argument('--dropout_rate', type=float, default=0.5, help='Float')
        parser.add_argument('--dueling', type=bool, default=True, help='Boolean')
        parser.add_argument('--epsilon_decrement', type=float, default=100, help='Float')
        parser.add_argument('--epsilon_end', type=float, default=0.01, help='Float')
        parser.add_argument('--gamma', type=float, default=0.99, help='Float')
        parser.add_argument('--learning_rate', type=float, default=1e-6, help='Float')
        parser.add_argument('--learning_rate_end', type=float, default=1e-7, help='Float')
        parser.add_argument('--scheduler_choice', type=str, default='constant', help='selected from https://pytorch.org/docs/stable/optim.html#algorithms, provided as a string')
        parser.add_argument('--lr_scheduler_epochs', type=int, default=2000, help='Integer')
        parser.add_argument('--optimizer', type=str, default='Adam', help='selected from https://pytorch.org/docs/stable/optim.html#algorithms, provided as a string')
        parser.add_argument('--loss_function', type=str, default='MSELoss', help='selected from https://pytorch.org/docs/stable/nn.html#loss-functions, provided as a string')
        parser.add_argument('--save_model_frequency', type=int, default=1000, help='Path to the hyperparameters file')
        parser.add_argument('--update_weight_percentage', type=float, default=1.0, help='Float')
        parser.add_argument('--memory_size', type=int, default=10000, help='Integer')
        parser.add_argument('--batch_size', type=int, default=32, help='Float')
        parser.add_argument('--replace_target_iter', type=int, default=3000, help='Float')
        
        parser.add_argument('--championship_windows', type=str, default=None, help='comma-separated integers')
        args = parser.parse_args()
        
        
        private_cpu_capacities = fill_array(args.private_cpu_capacities,args.number_of_servers,args.default_private_cpu_capacity,float)
        public_cpu_capacities = fill_array(args.public_cpu_capacities,args.number_of_servers,args.default_public_cpu_capacity,float)
        cloud_computational_capacity = args.cloud_computational_capacity
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
        epsilon_decrement = args.epsilon_decrement/(args.episode_time +max(timeout_delay_maxs))
        

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
                "cloud_computational_capacity":cloud_computational_capacity*args.time_step,
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
                "epsilon_decrement":epsilon_decrement,
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
                "championship_windows":comma_seperated_string_to_list(args.championship_windows,int)
        }
        
        json_object = json.dumps(hyperparameters,indent=4) ### this saves the array in .json format)

        
        with open(args.hyperparameters_file, 'w') as outfile:
            outfile.write(json_object)
if __name__ =="__main__":
    main()