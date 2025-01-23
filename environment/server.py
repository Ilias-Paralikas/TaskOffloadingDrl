import numpy as np
from .queues import ProcessingQueue,OffloadingQueue,PublicQueueManager


class Server():
    """
    A Server class that manages task processing, offloading and public queues in a distributed computing environment.
    This class represents a server node in a distributed system that can:
    - Process tasks locally in a private queue
    - Offload tasks to other connected servers
    - Handle tasks received from other servers in public queues
    - Track energy consumption and delays
    Attributes:
        id (int): Unique identifier for the server
        private_queue_computational_capacity (float): Processing capacity of server's private queue
        public_queues_computational_capacity (float): Processing capacity for handling tasks from other servers
        private_queue_waiting_time_consumption (float): Energy consumed while tasks wait in private queue
        private_queue_step_consumption (float): Energy consumed per processing step in private queue
        processing_queue (ProcessingQueue): Queue for processing local tasks
        offloading_servers (numpy.ndarray): Array of server IDs that this server can offload to
        offloading_capacities (dict): Mapping of server IDs to their offloading capacities
        offloading_queue_waiting_time_consumption (float): Energy consumed while tasks wait for offloading
        offloading_queue_step_consumption (float): Energy consumed per step during offloading
        offloading_queue (OffloadingQueue): Queue for tasks being offloaded to other servers
        supporting_servers (numpy.ndarray): Array of server IDs that can offload to this server
        public_queue_waiting_time_consumption (float): Energy consumed while tasks wait in public queues
        public_queue_step_consumption (float): Energy consumed per step in public queues
        public_queue_manager (PublicQueueManager): Manages queues for tasks received from other servers
        current_time (int): Current time step in the simulation
    Methods:
        reset(): Resets the server state and all queues to initial conditions
        get_waiting_times(): Returns waiting times for processing and offloading queues
        add_offloaded_tasks(offloaded_tasks): Adds tasks received from other servers to public queues
        step(action, local_task): Processes one time step of server operations
        get_features(): Returns current state features of the server
        get_number_of_features(): Returns the number of state features
        get_number_of_actions(): Returns the number of possible actions
        get_offliading_servers(): Returns list of servers this can offload to
        get_active_queues(): Returns currently active public queues
        get_supporting_servers(): Returns list of servers that can offload to this one
    """
 
    def __init__(self, 
                 id :int, 
                 private_queue_computational_capacity :float,
                 public_queues_computational_capacity :float,
                 outbound_connections,
                 inbound_connections,
                 private_queue_waiting_time_consumption,
                 private_queue_step_consumption,
                 offloading_queue_waiting_time_consumption,
                 offloading_queue_step_consumption,
                 public_queue_waiting_time_consumption,
                 public_queue_step_consumption):
        self.id=id
        self.private_queue_computational_capacity = private_queue_computational_capacity
        self.public_queues_computational_capacity = public_queues_computational_capacity
        self.private_queue_waiting_time_consumption = private_queue_waiting_time_consumption
        self.private_queue_step_consumption = private_queue_step_consumption
        
        self.processing_queue = ProcessingQueue(self.private_queue_computational_capacity,
                                                self.private_queue_waiting_time_consumption,
                                                self.private_queue_step_consumption)
        
        outbound_connections = np.array(outbound_connections)
        self.offloading_servers = np.where(outbound_connections!=0)[0]
        self.offloading_capacities = {s:outbound_connections[s] for s in self.offloading_servers}
        self.offloading_queue_waiting_time_consumption = offloading_queue_waiting_time_consumption
        self.offloading_queue_step_consumption = offloading_queue_step_consumption
        self.offloading_queue = OffloadingQueue(offloading_capacities = self.offloading_capacities,
                                                waiting_time_consumption = self.offloading_queue_waiting_time_consumption,
                                                step_consumption = self.offloading_queue_step_consumption)

        inbound_connections = np.array(inbound_connections)
        self.supporting_servers =  np.where(inbound_connections!=0)[0]
        self.public_queue_waiting_time_consumption = public_queue_waiting_time_consumption
        self.public_queue_step_consumption = public_queue_step_consumption
        self.public_queue_manager = PublicQueueManager(id=self.id,
                                                       computational_capacity=  self.public_queues_computational_capacity,
                                                       supporting_servers= self.supporting_servers,
                                                       waiting_time_consumption=self.public_queue_waiting_time_consumption,
                                                       step_consumption=self.public_queue_step_consumption)
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
                
        local_reward,process_energy_consumption = self.processing_queue.step()
        transmited_task,offloaded_reward,offloading_energy_consumption  = self.offloading_queue.step()
        
        
        
        foreign_delay_rewards,foreign_energy_rewards =  self.public_queue_manager.step()
        total_delays=  foreign_delay_rewards
        total_delays[self.id] = local_reward + offloaded_reward
        
        
        local_energy_consumption = process_energy_consumption + offloading_energy_consumption
        total_energy_rewards = foreign_energy_rewards
        total_energy_rewards[self.id] = local_energy_consumption
        return transmited_task,total_delays,total_energy_rewards
    
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