import numpy as np
from .queues import PublicQueueManager
 
 
class Cloud:
    """
    A class representing a Cloud computing environment with multiple servers and task management capabilities.
    This class simulates a cloud computing infrastructure that can process offloaded tasks
    across multiple servers. It manages server allocation, task queuing, and execution through
    a public queue manager.
    Attributes:
        number_of_servers (int): The total number of available servers in the cloud
        computational_capacity (float): Processing power of each server
        current_time (int): Current simulation time step
        supporting_servers (numpy.ndarray): Array of available server IDs
        public_queue_manager (PublicQueueManager): Manager handling task queues and execution
    Args:
        number_of_servers (int): Number of servers to initialize in the cloud
        computational_capacity (float): Processing capacity per server
        waiting_time_consumption (float): Energy consumption rate while tasks wait
        step_consumption (float): Energy consumption rate during task execution
    Methods:
        reset(): Resets the cloud environment to initial state
        step(): Advances simulation by one time step and returns rewards
        add_offloaded_tasks(offloaded_tasks): Adds new tasks to the public queue
        get_features(): Returns current queue lengths as state features
        get_active_queues(): Returns currently active queue information
        get_supporting_servers(): Returns array of available server IDs
    """
    def __init__(self,
                 number_of_servers ,
                 computational_capacity,
                 waiting_time_consumption,
                 step_consumption):
        self.number_of_servers=  number_of_servers
        self.computational_capacity = computational_capacity
        
        self.current_time=0
        self.supporting_servers = np.arange(self.number_of_servers)
        self.public_queue_manager = PublicQueueManager(id=self.number_of_servers,
                                                       computational_capacity=  self.computational_capacity,
                                                       supporting_servers= self.supporting_servers,
                                                       waiting_time_consumption=waiting_time_consumption,
                                                       step_consumption=step_consumption)
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