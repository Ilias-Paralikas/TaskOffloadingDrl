from .server import Server
from .cloud import Cloud
from .task_generator import TaskGenerator
from .matchmaker import Matchmaker
from utils import merge_dicts,dict_to_array,remove_diagonal_and_reshape
import numpy as np
import torch
import math
class Environment():
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
                 number_of_clouds=1) -> None:
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
                                inbound_connections=get_column(self.connection_matrix,i)) 
                        for i in range(number_of_servers)]
       
        self.matchmakers = [Matchmaker(id=s.id,
                                       offloading_servers=s.get_offliading_servers())
                            for s in self.servers]
        self.cloud = Cloud(number_of_servers=number_of_servers,
                           computational_capacity=cloud_computational_capacity)
        
        
        self.number_of_task_features=  self.task_generators[0].generate().get_number_of_features()
        self.number_of_server_features = self.servers[0].get_number_of_features()
        self.number_of_features = self.number_of_task_features + self.number_of_server_features
        self.static_frequency = static_frequency
        self.static_counter = 0
        
        self.max_reward = max(drop_penalty_maxs)
        self.max_waiting_time = max(timeout_delay_maxs)
        self.get_task_features_maxs()
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
    
    def scale_rewards(self,reward):
        return reward/self.max_reward
    
    def get_task_features_maxs(self):
        self.feature_maxes = self.task_generators[0].get_maxs()
        for g in self.task_generators:
            np.maximum(self.feature_maxes, g.get_maxs(), out=self.feature_maxes)  # Compare and store the max values

    def scale_task_features(self,task_features):
        return task_features/self.feature_maxes
    def scale_waiting_times(self,waiting_times):
        return waiting_times/self.max_waiting_time
    def pack_observation(self):
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
        
        rewards = self.cloud.step()
        
        for server_id in range(self.number_of_servers):
            action = self.matchmakers[server_id].match_action(server_id,actions[server_id])
            self.add_action_info(action,server_id,self.tasks[server_id])
            transmited_task, server_reward = self.servers[server_id].step(action,self.tasks[server_id])
            rewards = merge_dicts(rewards,server_reward)
            if transmited_task:
                origin_server_id = transmited_task.get_origin_server_id()
                assert origin_server_id == server_id
                target_server_id = transmited_task.get_target_server_id()
                
                self.horisontal_transmitted_tasks[target_server_id].append(transmited_task) 

        self.tasks= [t.step() for t in self.task_generators]     
               
        observations = self.pack_observation()
        rewards  = dict_to_array(rewards,self.number_of_servers)
        rewards = self.scale_rewards(rewards)
        rewards = -rewards
        
        # tasks_dropped = 
        
        info  ={}
        info['rewards'] = rewards
        info['tasks_arrived'] = np.array(tasks_arrived)
        info['tasks_dropped'] = -np.ceil(rewards)
        
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
        
        
        