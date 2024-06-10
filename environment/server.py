import numpy as np
from .queues import ProcessingQueue,OffloadingQueue,PublicQueueManager


class Server():
    def __init__(self, 
                 id :int, 
                 private_queue_computational_capacity :float,
                 public_queues_computational_capacity :float,
                 connection_row :np.array,):
        self.id=id
        self.private_queue_computational_capacity = private_queue_computational_capacity
        self.public_queues_computational_capacity = public_queues_computational_capacity
        self.connection_row = connection_row
       
        self.supporting_servers =  np.where(connection_row!=0)[0]
        
        self.processing_queue = ProcessingQueue(self.private_queue_computational_capacity)
        
        self.offloading_queue = OffloadingQueue(offloading_capacities = self.connection_row)

        self.public_queue_manager = PublicQueueManager(id=self.id,
                                                       computational_capacity=  self.public_queues_computational_capacity,
                                                       supporting_servers= self.supporting_servers)
        self.current_time=0

    def reset(self):
            self.current_time=0
            self.processing_queue.reset()
            self.public_queue_manager.reset()
            self.offloading_queue.reset()   
    
    def get_waiting_times(self):
        return  self.processing_queue.get_waiting_time(),self.offloading_queue.get_waiting_time()
    
    def add_offloaded_tasks(self,offloaded_tasks):
        self.public_queue_manager.add_tasks(offloaded_tasks)

    def step(self,action=None,local_task=None):
        if local_task:
            local_task.set_origin_server_id(self.id)
            if action ==self.id:  
                self.processing_queue.add_task(local_task)
            else:
                target_server_id = action
                local_task.set_target_server_id(target_server_id)
                self.offloading_queue.add_task(local_task)
                
        local_reward = self.processing_queue.step()
        transmited_task,offloaded_reward  = self.offloading_queue.step()
        foreign_rewards =  self.public_queue_manager.step()
        total_rewards= local_reward + offloaded_reward + foreign_rewards
        return transmited_task,total_rewards