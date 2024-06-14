from .server import Server
from .cloud import Cloud
from .task_generator import TaskGenerator
from .matchmaker import Matchmaker
from utils import merge_dicts,dict_to_array,remove_diagonal_and_reshape
import numpy as np


class Environment():
    def __init__(self, 
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
        
    def reset(self):
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
        return observations,done, info
    def reset_transmitted_tasks(self):
        self.horisontal_transmitted_tasks = [[] for _ in range(self.number_of_servers+self.number_of_clouds)]
    
        self.tasks = [t.step() for t in self.task_generators]
        
    def pack_observation(self):
        local_observations = np.zeros((self.number_of_servers,self.number_of_features))
        public_queues  = [np.array([]) for key in range(self.number_of_servers+self.number_of_clouds)]
        assert len(self.tasks) == self.number_of_servers
        for s in range(self.number_of_servers):
            if self.tasks[s]:
                task_features = self.tasks[s].get_features()
            else:
                task_features = np.zeros(self.number_of_task_features)
            waiting_times,server_public_queues = self.servers[s].get_features()            
            server_features = np.concatenate([task_features,waiting_times])
            local_observations[s] = server_features
        
            for q in server_public_queues:
                public_queues[q] = np.append(public_queues[q], server_public_queues[q])

        cloud_public_queues = self.cloud.get_features()
        for q in cloud_public_queues:
            public_queues[q] = np.append(public_queues[q], cloud_public_queues[q])      
        return local_observations,public_queues
    def step(self,actions):
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
        
        info  ={}
        info['rewards'] = rewards
        return observations,rewards, done, info
        
        
    def get_server_dimensions(self,id):
        return (self.servers[id].get_number_of_features(),
                self.servers[id].get_number_of_actions()-1,
                self.servers[id].get_number_of_actions()
        )
    def get_task_features(self):
        return self.task_generators[0].get_number_of_features()