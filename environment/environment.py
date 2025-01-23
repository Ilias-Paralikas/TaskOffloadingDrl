from .server import Server
from .cloud import Cloud
from .task_generator import TaskGenerator
from .matchmaker import Matchmaker
from utils import merge_dicts,dict_to_array,remove_diagonal_and_reshape
import numpy as np
import torch
import math

class Environment():
    """
    A class representing the task offloading environment for distributed computing systems.
    This environment simulates a distributed computing system where multiple servers can process tasks
    locally, offload them to other servers horizontally, or send them to a cloud server. The environment
    handles task generation, server management, and resource allocation.
    Parameters
    ----------
    static_frequency : int
        Frequency of resetting random seeds for reproducibility
    number_of_servers : int 
        Number of edge servers in the system
    private_cpu_capacities : list
        CPU capacities for private queues of each server
    public_cpu_capacities : list 
        CPU capacities for public queues of each server
    connection_matrix : 2D array
        Matrix defining connectivity between servers (1 if connected, 0 if not)
    cloud_computational_capacity : float
        Computational capacity of the cloud server
    episode_time : int
        Duration of each episode
    task_arrive_probabilities : list
        Probability of task arrival for each server
    task_size_mins : list
        Minimum task sizes for each server
    task_size_maxs : list
        Maximum task sizes for each server 
    task_size_distributions : list
        Probability distributions for task sizes
    timeout_delay_mins : list
        Minimum timeout delays for each server
    timeout_delay_maxs : list
        Maximum timeout delays for each server
    timeout_delay_distributions : list
        Probability distributions for timeout delays
    priotiry_mins : list
        Minimum priority values for tasks
    priotiry_maxs : list
        Maximum priority values for tasks
    priotiry_distributions : list
        Probability distributions for task priorities
    computational_density_mins : list
        Minimum computational density for tasks
    computational_density_maxs : list
        Maximum computational density for tasks
    computational_density_distributions : list
        Probability distributions for computational densities
    drop_penalty_mins : list
        Minimum penalties for dropping tasks
    drop_penalty_maxs : list
        Maximum penalties for dropping tasks
    drop_penalty_distributions : list
        Probability distributions for drop penalties
    private_queue_waiting_time_consumptions : list
        Energy consumption rates for waiting in private queues
    private_queue_step_consumptions : list
        Energy consumption per step in private queues
    public_queue_waiting_time_consumptions : list
        Energy consumption rates for waiting in public queues
    public_queue_step_consumptions : list
        Energy consumption per step in public queues
    offloading_queue_waiting_time_consumptions : list
        Energy consumption rates for waiting in offloading queues
    offloading_queue_step_consumptions : list
        Energy consumption per step in offloading queues
    cloud_waiting_time_consumption : float
        Energy consumption rate for waiting in cloud queue
    cloud_step_consumption : float
        Energy consumption per step in cloud
    delay_to_energy_ratio : float
        Weight ratio between delay and energy in reward calculation
    number_of_clouds : int, optional
        Number of cloud servers (default is 1)
    scale_iterations : int, optional
        Number of iterations for calculating reward scaling factors (default is 100)
    Attributes
    ----------
    number_of_servers : int
        Number of edge servers in the system
    number_of_clouds : int
        Number of cloud servers
    current_time : int
        Current timestep in the episode
    episode_time_end : int
        End time of the episode
    connection_matrix : 2D array
        Connectivity matrix between servers
    task_generators : list
        List of TaskGenerator objects for each server
    servers : list
        List of Server objects
    matchmakers : list
        List of Matchmaker objects for task allocation
    cloud : Cloud
        Cloud server object
    number_of_task_features : int
        Number of features describing each task
    number_of_server_features : int
        Number of features describing each server
    number_of_features : int
        Total number of features (task + server features)
    static_frequency : int
        Frequency of random seed resets
    max_waiting_time : float
        Maximum allowed waiting time for tasks
    timeout_penalty : float
        Maximum penalty for task timeout
    delay_to_energy_ratio : float
        Weight between delay and energy in reward calculation
    Methods
    -------
    reset()
        Resets the environment to initial state
    step(actions)
        Executes one timestep in the environment given actions
    pack_observation()
        Creates observation vector for current state
    get_server_dimensions(id)
        Returns dimensions of observation/action spaces for a server
    get_task_features()
        Returns number of task features
    get_episode_actions()
        Returns statistics about actions taken in episode
    get_foreign_cpus(id)
        Returns available CPU capacities for offloading
    """
    def __init__(self, 
                 static_frequency,
                 number_of_servers,
                 private_cpu_capacities,
                 public_cpu_capacities,
                 connection_matrix,
                 cloud_computational_capacity,
                 episode_time,
                 task_arrive_probabilities,
                 task_size_mins,
                 task_size_maxs,
                 task_size_distributions,
                timeout_delay_mins,
                timeout_delay_maxs,
                timeout_delay_distributions,
                priotiry_mins,
                priotiry_maxs,
                priotiry_distributions,
                computational_density_mins,
                computational_density_maxs,
                computational_density_distributions,
                drop_penalty_mins,
                drop_penalty_maxs,
                drop_penalty_distributions, 
                private_queue_waiting_time_consumptions,
                private_queue_step_consumptions,
                public_queue_waiting_time_consumptions,
                public_queue_step_consumptions,
                offloading_queue_waiting_time_consumptions,
                offloading_queue_step_consumptions,
                cloud_waiting_time_consumption,
                cloud_step_consumption,
                delay_to_energy_ratio, 
                 number_of_clouds=1,
                 scale_iterations=100) -> None:
        self.number_of_servers = number_of_servers
        self.number_of_clouds = number_of_clouds
        self.current_time = 0
        self.episode_time_end = episode_time +max(timeout_delay_maxs)
        self.connection_matrix=  connection_matrix
        get_column = lambda m, i: [row[i] for row in m]
        self.task_generators = [TaskGenerator(id=i,
                                              episode_time=episode_time,
                                              task_arrive_probability=task_arrive_probabilities[i],
                                              size_min=task_size_mins[i],
                                              size_max = task_size_maxs[i],
                                                size_distribution = task_size_distributions[i],
                                                timeout_delay_min = timeout_delay_mins[i],
                                                timeout_delay_max = timeout_delay_maxs[i],
                                                timeout_delay_distribution = timeout_delay_distributions[i],
                                                priotiry_min = priotiry_mins[i],
                                                priotiry_max = priotiry_maxs[i],
                                                priotiry_distribution = priotiry_distributions[i],
                                                computational_density_min = computational_density_mins[i],
                                                computational_density_max = computational_density_maxs[i],
                                                computational_density_distribution = computational_density_distributions[i],
                                                drop_penalty_min = drop_penalty_mins[i],
                                                drop_penalty_max = drop_penalty_maxs[i],
                                                drop_penalty_distribution = drop_penalty_distributions[i])
                                for i in range(number_of_servers)]         
        self.servers = [Server( id=i,
                                private_queue_computational_capacity=  private_cpu_capacities[i],
                                public_queues_computational_capacity= public_cpu_capacities[i],
                                outbound_connections=  self.connection_matrix[i],
                                inbound_connections=get_column(self.connection_matrix,i),
                                private_queue_waiting_time_consumption=private_queue_waiting_time_consumptions[i],
                                private_queue_step_consumption=private_queue_step_consumptions[i],
                                offloading_queue_waiting_time_consumption=offloading_queue_waiting_time_consumptions[i],
                                offloading_queue_step_consumption=offloading_queue_step_consumptions[i],
                                public_queue_waiting_time_consumption=public_queue_waiting_time_consumptions[i],
                                public_queue_step_consumption=public_queue_step_consumptions[i]) 
                        for i in range(number_of_servers)]
       
        self.matchmakers = [Matchmaker(id=s.id,
                                       offloading_servers=s.get_offliading_servers())
                            for s in self.servers]
        self.cloud = Cloud(number_of_servers=number_of_servers,
                           computational_capacity=cloud_computational_capacity,
                           waiting_time_consumption = cloud_waiting_time_consumption,
                            step_consumption = cloud_step_consumption)
        
        
        self.number_of_task_features=  self.task_generators[0].generate().get_number_of_features()
        self.number_of_server_features = self.servers[0].get_number_of_features()
        self.number_of_features = self.number_of_task_features + self.number_of_server_features
        self.static_frequency = static_frequency
        self.static_counter = 0
        
        self.max_waiting_time = max(timeout_delay_maxs)
        self.timeout_penalty = max(drop_penalty_maxs)

        
        self.delay_to_energy_ratio= delay_to_energy_ratio
        self.get_task_features_maxs()
        self.get_scaling_factors(scale_iterations)
        self.reset()
    def reset(self):
        
        
        
        if self.static_frequency:
            if self.static_counter % self.static_frequency ==0:
                np.random.seed(42)
                torch.manual_seed(42)
                self.static_counter+=1
        self.current_time = 0
        for task_generator in self.task_generators:
            task_generator.reset()
        for server in self.servers:
            server.reset()
        self.cloud.reset()
        self.reset_transmitted_tasks()
        self.tasks= [t.step() for t in self.task_generators]
        
        observations = self.pack_observation()
        done = False
        info = {}
        self.actions  =[{
                'local':0,
                'horisontal':0,
                'cloud':0
            }
            for _ in range(self.number_of_servers)]
        return observations,done, info
    def reset_transmitted_tasks(self):
        self.horisontal_transmitted_tasks = [[] for _ in range(self.number_of_servers+self.number_of_clouds)]
    
  
    
    
    def get_task_features_maxs(self):
        self.feature_maxes = self.task_generators[0].get_maxs()
        for g in self.task_generators:
            np.maximum(self.feature_maxes, g.get_maxs(), out=self.feature_maxes)  # Compare and store the max values

    def scale_task_features(self,task_features):
        return task_features/self.feature_maxes
    def scale_waiting_times(self,waiting_times):
        return waiting_times/self.max_waiting_time
    def pack_observation(self):
        """
        Packs observation data from servers, tasks, and cloud into a structured format for the environment.
        This function collects and processes various features from the environment components:
        - Gathers task features and waiting times from each server
        - Collects public queue lengths from servers and cloud
        - Builds local observations by combining server observations with their public queue lengths
        - Tracks active queues and their supporting server relationships
        Returns:
            tuple: Contains two elements:
                - local_observations (list): List of numpy arrays where each array contains:
                    * Scaled task features (if task exists, zeros if no task)
                    * Scaled waiting times for the server
                    * Public queue lengths for that server
                - active_queues (list): List of numpy arrays containing active queue information
                    for each server, including data from supporting servers and cloud
        Structure Details:
            - server_observations: Matrix of size (number_of_servers × number_of_features)
            - public_queues_length: List of arrays tracking queue lengths for servers and clouds
            - active_queues: List of arrays showing which queues are currently active for each server
        Notes:
            - Task features and waiting times are scaled before being packed
            - Empty tasks are represented by zero vectors
            - Cloud queues are processed separately and added to relevant servers' observations
        Requires:
            - self.tasks and self.servers to be properly initialized
            - All component classes (Server, Cloud, Task) to implement proper get_features() methods
        """
        server_observations = np.zeros((self.number_of_servers,self.number_of_features))
        public_queues_legth  = [np.array([]) for key in range(self.number_of_servers+self.number_of_clouds)]
        assert len(self.tasks) == self.number_of_servers
        for s in range(self.number_of_servers):
            if self.tasks[s]:
                task_features = self.tasks[s].get_features()
            else:
                task_features = np.zeros(self.number_of_task_features)
            task_features = self.scale_task_features(task_features)
            waiting_times,server_public_queues = self.servers[s].get_features()     
            waiting_times = self.scale_waiting_times(waiting_times)       
            server_features = np.concatenate([task_features,waiting_times])
            server_observations[s] = server_features
        
            for q in server_public_queues:
                public_queues_legth[q] = np.append(public_queues_legth[q], server_public_queues[q])

        cloud_public_queues = self.cloud.get_features()
        for q in cloud_public_queues:
            public_queues_legth[q] = np.append(public_queues_legth[q], cloud_public_queues[q])      
            
        local_observations = []
        for i in range(len(server_observations)):
            local_observations.append(np.append(server_observations[i],public_queues_legth[i]))

        active_queues = [np.array([]) for _ in range(self.number_of_servers) ]
        for s in self.servers:
            server_active_queues = s.get_active_queues()
            supporting_servers  = s.get_supporting_servers()
            for target_server in supporting_servers:
                active_queues[target_server] = np.append( active_queues[target_server],server_active_queues)
            
        cloud_active_queeus = self.cloud.get_active_queues()
        supporting_servers  = self.cloud.get_supporting_servers()

        for target_server in supporting_servers:
            active_queues[target_server] = np.append( active_queues[target_server],cloud_active_queeus)
            
        return local_observations,active_queues
    
    def add_action_info(self,action,server_id,task):
        if task:
            if action ==server_id:
                self.actions[server_id]['local'] +=1
            elif action == self.number_of_servers:
                self.actions[server_id]['cloud'] +=1
            else:
                self.actions[server_id]['horisontal'] +=1
                
    def step(self,actions):
        """
        Execute a step in the task offloading environment simulation.
        This method advances the simulation by one time step, processing task offloading decisions,
        calculating rewards, and updating the system state.
        Parameters
        ----------
        actions : list or numpy.ndarray
            List of actions for each server, where len(actions) equals number_of_servers.
            Each action determines how the server should handle its current task.
        Returns
        -------
        tuple
            Contains 4 elements:
            - observations (dict): Current state of the environment after the step
            - rewards (numpy.ndarray): Combined rewards (delay + energy) for each server
            - done (bool): True if episode has ended (current_time >= episode_time_end)
            - info (dict): Additional information containing, to be sent to the bookkeeper:
                - delay_rewards: Raw delay penalties for each server
                - delay_without_drop_rewards: Delay penalties excluding dropped tasks
                - energy_rewards: Energy consumption penalties for each server
                - rewards: Scaled and combined rewards
                - tasks_arrived: Binary array indicating task arrival at each server
                - tasks_dropped: Number of dropped tasks per server
        Notes
        -----
        The step function performs the following operations:
        1. Processes horizontally transmitted tasks between servers
        2. Executes cloud server step
        3. Processes each edge server's actions and calculates rewards
        4. Handles task transmissions between servers
        5. Generates new tasks
        6. Updates environment state and scales rewards
        The rewards are calculated as a weighted sum of delay and energy penalties,
        controlled by delay_to_energy_ratio parameter. Both delay and energy rewards
        are scaled using their respective scaling factors before combination.
        """

        tasks_arrived = [0 if t is None else 1 for t in self.tasks]
        
        assert len(actions) == self.number_of_servers
        if self.current_time >=self.episode_time_end:
            done = True
        else:
            done = False
        self.current_time +=1
        
        for s in self.servers:
            s.add_offloaded_tasks(self.horisontal_transmitted_tasks[s.id])
        self.cloud.add_offloaded_tasks(self.horisontal_transmitted_tasks[-1])
        self.reset_transmitted_tasks()
        
        delay_rewards,energy_rewards = self.cloud.step()
        
        for server_id in range(self.number_of_servers):
            action = self.matchmakers[server_id].match_action(server_id,actions[server_id])
            self.add_action_info(action,server_id,self.tasks[server_id])
            transmited_task, server_delay_reward,server_energey_rewards = self.servers[server_id].step(action,self.tasks[server_id])
            delay_rewards = merge_dicts(delay_rewards,server_delay_reward)
            energy_rewards = merge_dicts(energy_rewards,server_energey_rewards)
            if transmited_task:
                origin_server_id = transmited_task.get_origin_server_id()
                assert origin_server_id == server_id
                target_server_id = transmited_task.get_target_server_id()
                
                self.horisontal_transmitted_tasks[target_server_id].append(transmited_task) 

        self.tasks= [t.step() for t in self.task_generators]     
               
        observations = self.pack_observation()
        
        delay_rewards  = dict_to_array(delay_rewards,self.number_of_servers)
        delay_rewards = -delay_rewards
        scaled_delay_rewards = delay_rewards/self.delay_scaling
        
        energy_rewards = dict_to_array(energy_rewards,self.number_of_servers)
        energy_rewards = -energy_rewards
        scaled_energy_rewards = energy_rewards/self.energy_scaling

        
        rewards =   self.delay_to_energy_ratio *scaled_delay_rewards +  \
                    (1-self.delay_to_energy_ratio) *scaled_energy_rewards
        info  ={}
        
        tasks_dropped  =-np.ceil(delay_rewards/self.timeout_penalty)
        delay_without_drop = delay_rewards + tasks_dropped*self.timeout_penalty
        
        info['delay_rewards'] = delay_rewards
        info['delay_without_drop_rewards'] = delay_without_drop
        info['energy_rewards'] = energy_rewards
        info['rewards'] = rewards
        info['tasks_arrived'] = np.array(tasks_arrived)
        info['tasks_dropped'] = tasks_dropped
        
        return observations,rewards, done, info
        
    
    def get_server_dimensions(self,id):
        
        local_observations, active_queues = self.pack_observation()
        local_observations = local_observations[id]
        active_queues  =  active_queues[id]
        return (len(local_observations),
                len(active_queues),
                self.servers[id].get_number_of_actions()
        )
    def get_task_features(self):
        return self.task_generators[0].get_number_of_features()
    
    def get_episode_actions(self):
        return self.actions
    
    def get_foreign_cpus(self,id):
        available_servers = np.array(self.matchmakers[id].get_rows())
        available_servers = available_servers[available_servers!=id]
        available_servers = available_servers[available_servers!=self.number_of_servers]
        public_cpus = np.array([s.public_queues_computational_capacity for s in self.servers])
        
        available_public_cpus = public_cpus[available_servers]
        
        available_public_cpus = np.append(available_public_cpus, self.cloud.computational_capacity)
        return available_public_cpus
        
        
        
        
    def get_scaling_factors(self,iterations):
        delay_rewards = []
        energy_rewards = []
        self.delay_scaling = 1
        self.energy_scaling = 1
        
        for _ in range(iterations):
            episode_delay_rewards = []
            episode_energy_rewards = []
            self.reset()

            for _ in range(self.episode_time_end):
                actions = [np.random.randint(0,self.servers[j].get_number_of_actions()) for j in range(self.number_of_servers)]
                _, _,_,info =self.step(actions)
                mean_delay_rewards = np.mean(info['delay_rewards'])
                mean_energy_rewards = np.mean(info['energy_rewards'])
                
                episode_delay_rewards.append(mean_delay_rewards)
                episode_energy_rewards.append(mean_energy_rewards)
            
            delay_rewards.append(np.sum(episode_delay_rewards))
            energy_rewards.append(np.sum(episode_energy_rewards))
            
        self.delay_scaling =  - np.mean(delay_rewards)
        self.energy_scaling =  - np.mean(energy_rewards)
    