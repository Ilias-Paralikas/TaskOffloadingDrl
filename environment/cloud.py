import numpy as np
from .queues import PublicQueueManager
 
 
class Cloud:
    def __init__(self,
                 number_of_servers ,
                 computational_capacity):
        self.number_of_servers=  number_of_servers
        self.computational_capacity = computational_capacity
        
        self.current_time=0
        self.supporting_servers = np.arange(self.number_of_servers)
        self.public_queue_manager = PublicQueueManager(id=self.number_of_servers,
                                                       computational_capacity=  self.computational_capacity,
                                                       supporting_servers= self.supporting_servers)
    def reset(self):
        self.current_time=0
        self.public_queue_manager.reset()

    
    def step(self):
        rewards = self.public_queue_manager.step()
        return rewards
    
    def add_offloaded_tasks(self,offloaded_tasks):
        self.public_queue_manager.add_tasks(offloaded_tasks)

    def get_features(self):
        return self.public_queue_manager.get_queue_lengths()
    
 
    def get_active_queues(self):
        active_queues =self.public_queue_manager.get_active_queues()
        return active_queues
    
    def get_supporting_servers(self):
        return self.supporting_servers