import numpy as np
from .queues import ProcessingQueue,OffloadingQueue,PublicQueueManager


class Server():
    def __init__(self, 
                 id :int, 
                 private_queue_computational_capacity :float,
                 public_queues_computational_capacity :float,
                 outbound_connections,
                 inbound_connections):
        self.id=id
        self.private_queue_computational_capacity = private_queue_computational_capacity
        self.public_queues_computational_capacity = public_queues_computational_capacity
       
        
        self.processing_queue = ProcessingQueue(self.private_queue_computational_capacity)
        
        outbound_connections = np.array(outbound_connections)
        self.offloading_servers = np.where(outbound_connections!=0)[0]
        self.offloading_capacities = {s:outbound_connections[s] for s in self.offloading_servers}
        self.offloading_queue = OffloadingQueue(offloading_capacities = self.offloading_capacities)

        inbound_connections = np.array(inbound_connections)
        self.supporting_servers =  np.where(inbound_connections!=0)[0]
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
        total_rewards=  foreign_rewards
        total_rewards[self.id] = local_reward + offloaded_reward
        return transmited_task,total_rewards
    
    def get_features(self):
        private_waiting_time,public_waiting_time = self.get_waiting_times()
        public_queues = self.public_queue_manager.get_queue_lengths()
        return np.array([private_waiting_time,
                         public_waiting_time]),public_queues
    
    def get_number_of_features(self):
        features,_  = self.get_features()
        return len(features)
    def get_number_of_actions(self):
        return 1+len(self.offloading_servers)
    
    def get_offliading_servers(self):
        return self.offloading_servers
    
    
    def get_active_queues(self):
        active_queues  =self.public_queue_manager.get_active_queues()
        return active_queues
    
    
    def get_supporting_servers(self):
        return self.supporting_servers