from .task import Task
from utils import merge_dicts
import queue
import math
class TaskQueue():
    def __init__(self)->None:
        self.reset()
        
    def reset(self)->None:
        self.current_time=0
        self.queue_length = 0
        self.queue = queue.Queue()
        self.current_task = Task()
    
    def add_task(self,
                 task:Task)->None:
        if self.is_empty():
            self.current_task = task.copy()
        else:
            self.queue.put(task.copy())
        self.queue_length += task.get_size()
        
    def is_empty(self) -> bool:
        return self.queue.empty() and self.current_task.is_empty() 
    
    def current_task_is_timed_out(self)->bool:
        return self.current_time >= self.current_task.get_timeout()

    def get_first_non_empty_element(self) ->int:
        self.current_time +=1
        rewards =  0
        if self.current_task.is_empty():
            if self.queue.empty():
                return rewards
            else:
                self.current_task = self.queue.get()
        while self.current_task_is_timed_out():
            self.queue_length -= self.current_task.get_remaining_size()
            rewards += self.current_task.drop_task()
            if self.queue.empty():
                return rewards
            else:
                self.current_task = self.queue.get()
        return rewards
    
    def get_waiting_time(self):
        try:
            return self.waiting_time
        except:
            raise NotImplementedError("Waiting time is not implemented for this queue (Probably public queue)")
    def get_queue_length(self):
        return self.queue_length

class ProcessingQueue(TaskQueue):
    def __init__(self,computational_capacity):
        super().__init__()
        self.computational_capacity = computational_capacity
        self.waiting_time =0
    def reset(self):
        super().reset()
        self.waiting_time =0      
    
    def update_waiting_time(self,task):
        process_per_time_period = self.computational_capacity / task.get_density()
        time_to_process_task = math.ceil(task.get_size()/process_per_time_period)
        timeout_time = max(0,task.get_relative_timeout()-self.waiting_time)
        self.waiting_time += min(timeout_time,time_to_process_task)
        
    def add_task(self,task):
        super().add_task(task)
        self.update_waiting_time(task)
        
        
    def step(self):
        if self.waiting_time>0:
            self.waiting_time -=1
        rewards = self.get_first_non_empty_element()
        if self.current_task.is_empty():
            return rewards
        rewards += self.current_task.process(self.computational_capacity,self.current_time)
        return rewards
    
    

class OffloadingQueue(TaskQueue):
    def __init__(self,offloading_capacities):   
        super().__init__()
        self.offloading_capacities = offloading_capacities
        self.reset()
    def reset(self):
        super().reset()
        self.waiting_time =0
    
    def update_waiting_time(self, task):
        target_server_id = task.get_target_server_id()
        offloading_capacity = self.offloading_capacities[target_server_id]
        time_to_transmit_task =  math.ceil(task.get_size()/ offloading_capacity)
        timeout_time = max(0,task.get_relative_timeout()- self.waiting_time)
        self.waiting_time += min(timeout_time,time_to_transmit_task)
        
    
    def add_task(self,task):
        super().add_task(task)
        self.update_waiting_time(task)
        
    def step(self):
        if self.waiting_time>0:
            self.waiting_time -=1
        transmited_task = None
        reward = self.get_first_non_empty_element()
        if self.current_task.is_empty():
            return transmited_task,reward

        target_server_id = self.current_task.get_target_server_id()
        offloading_capacity = self.offloading_capacities[target_server_id]
        transmited_task = self.current_task.transmit(offloading_capacity)
        return transmited_task,reward
    
class PublicQueue(TaskQueue):
    def step(self,computational_capacity):
        reward = 0
        if self.current_task.is_empty():
            return reward
        reward, task_processed = self.current_task.public_process(computational_capacity,self.current_time)
        self.queue_length -= task_processed
        return reward
                    

class PublicQueueManager():
    def __init__(self,
                 id,
                 computational_capacity,
                 supporting_servers):
        self.id = id
        self.computational_capacity = computational_capacity
        self.supporting_servers = supporting_servers
        self.public_queues ={}
        for server_id in self.supporting_servers:
            self.public_queues[server_id] = PublicQueue()
       
    
    def reset(self):
        for _,q in self.public_queues.items():
            q.reset()
    
    def get_public_queue_server_length(self,server_id):
        return self.public_queues[server_id].queue_length
    
    def get_priorities(self):
        total_priority =0
        for _,q in self.public_queues.items():
            if not q.is_empty():
                total_priority += q.current_task.get_priority() 
        return total_priority
    def get_active_queues(self):
        active_queues =0
        for _,q in self.public_queues.items():
            if not q.is_empty():
                active_queues +=1
        return active_queues
        
    def add_tasks(self,recieved_tasks=[]):
        for task in recieved_tasks:
            assert task.get_target_server_id() == self.id
            origin_server_id = task.get_origin_server_id()
            self.public_queues[origin_server_id].add_task(task)
    
    def step(self):
        drop_rewards ={}
        for server_id in self.supporting_servers:
            drop_rewards[server_id]= self.public_queues[server_id].get_first_non_empty_element()
        
        finished_rewards = {}
        active_queues= self.get_active_queues()
        total_priority = self.get_priorities()
        if active_queues!=0:
            distributed_computational_capacity = self.computational_capacity/total_priority
        else:
            distributed_computational_capacity = 0
       
        for server_id in self.supporting_servers:
            finished_rewards[server_id]= self.public_queues[server_id].step(distributed_computational_capacity)
         
        rewards = merge_dicts(drop_rewards,finished_rewards)
        return rewards
    
    
    def get_queue_lengths(self):
        queue_lengths = {}
        for server_id in self.supporting_servers:
            queue_lengths[server_id] = self.public_queues[server_id].get_queue_length()
        return queue_lengths
    