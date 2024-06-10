from .server import Server
from .cloud import Cloud
from .task_generator import TaskGenerator
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
        self.servers = [Server(id=i,
                               private_queue_computational_capacity=  private_cpu_capacities[i],
                            public_queues_computational_capacity= public_cpu_capacities[i],
                                connection_row=  connection_matrix[i]) 
                        for i in range(number_of_servers)]
        self.cloud = Cloud(number_of_servers=number_of_servers,
                           computational_capacity=cloud_computational_capacity)
        
        self.reset()
    def reset(self):
        self.current_time = 0
        for task_generator in self.task_generators:
            task_generator.reset()
        for server in self.servers:
            server.reset()
        self.cloud.reset()
        self.reset_transmitted_tasks()
        self.tasks= [t.step() for t in self.task_generators]
    def reset_transmitted_tasks(self):
        self.horisontal_transmitted_tasks = [[] for _ in range(self.number_of_servers+self.number_of_clouds)]
    
        
    def step(self,actions):
        assert len(actions) == self.number_of_servers
        if self.current_time >=self.episode_time_end:
            done = True
        else:
            done = False
        self.current_time +=1
        
        rewards = []
        for s in self.servers:
            s.add_offloaded_tasks(self.horisontal_transmitted_tasks[s.id])
        self.cloud.add_offloaded_tasks(self.horisontal_transmitted_tasks[-1])
        self.reset_transmitted_tasks()
        
        
        for server_id in range(self.number_of_servers):
            transmited_task, server_reward = self.servers[server_id].step(actions[server_id],self.tasks[server_id])
            rewards += server_reward
            if transmited_task:
                origin_server_id = transmited_task.get_origin_server_id()
                assert origin_server_id == server_id
                target_server_id = transmited_task.get_target_server_id()
                
                self.horisontal_transmitted_tasks[target_server_id].append(transmited_task) 

        self.tasks= [t.step() for t in self.task_generators]
        
        
            
               
            
        return rewards, done    
        
        