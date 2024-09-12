from .task import Task
from utils import merge_dicts
import queue
import math
class TaskQueue():
    def __init__(self,
                 waiting_time_consumption,
                 step_consumption)->None:
        self.waiting_time_consumption=  waiting_time_consumption
        self.step_consumption = step_consumption
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

    def calculate_energy_consumption(self,task_size):
        
        process_energy_consumption = task_size * self.step_consumption
        waiting_time_energy_consumption = self.queue_length * self.waiting_time_consumption
        energy_consumption = process_energy_consumption + waiting_time_energy_consumption
                
        return energy_consumption
class ProcessingQueue(TaskQueue):
    def __init__(self,
                 computational_capacity,
                waiting_time_consumption,
                 step_consumption):
        super().__init__(waiting_time_consumption,step_consumption)
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
        energy_consumption =0
        if self.waiting_time>0:
            self.waiting_time -=1
        rewards = self.get_first_non_empty_element()
        if self.current_task.is_empty():
            return rewards,energy_consumption
        process_reward,task_processed,computational_density= self.current_task.process(self.computational_capacity,self.current_time)
        rewards += process_reward
        
        self.queue_length -= task_processed
        energy_consumption = self.calculate_energy_consumption(task_processed *computational_density)
        
        return rewards,energy_consumption
    
    

class OffloadingQueue(TaskQueue):
    def __init__(self,
                 offloading_capacities,
                 waiting_time_consumption,
                 step_consumption):   
        super().__init__(waiting_time_consumption,step_consumption)
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
        energy_consumption =0
        if self.waiting_time>0:
            self.waiting_time -=1
        transmitted_task = None
        reward = self.get_first_non_empty_element()
        if self.current_task.is_empty():
            return transmitted_task,reward,energy_consumption


        target_server_id = self.current_task.get_target_server_id()
        offloading_capacity = self.offloading_capacities[target_server_id]
        transmitted_task,transmitted_size = self.current_task.transmit(offloading_capacity)
        
        
        self.queue_length -= transmitted_size
        energy_consumption = self.calculate_energy_consumption(transmitted_size)

        
        return transmitted_task,reward,energy_consumption
    
class PublicQueue(TaskQueue):
    def step(self,computational_capacity):
        rewards = 0
        energy_consumption = 0
        if self.current_task.is_empty():
            return rewards,energy_consumption
        rewards, task_processed,computational_density = self.current_task.public_process(computational_capacity,self.current_time)
        
        self.queue_length -= task_processed
        energy_consumption = self.calculate_energy_consumption(task_processed*computational_density)

        return rewards,energy_consumption
    
        
                    

class PublicQueueManager():
    def __init__(self,
                 id,
                 computational_capacity,
                 supporting_servers,
                 waiting_time_consumption,
                 step_consumption):
        self.id = id
        self.computational_capacity = computational_capacity
        self.supporting_servers = supporting_servers
        self.public_queues ={}
        for server_id in self.supporting_servers:
            self.public_queues[server_id] = PublicQueue(waiting_time_consumption,
                                                        step_consumption)
       
    
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
        energy_rewards ={}
        active_queues= self.get_active_queues()
        total_priority = self.get_priorities()
        if active_queues!=0:
            distributed_computational_capacity = self.computational_capacity/total_priority
        else:
            distributed_computational_capacity = 0
       
        for server_id in self.supporting_servers:
            finished_rewards[server_id],energy_rewards[server_id]= self.public_queues[server_id].step(distributed_computational_capacity)
         
        rewards = merge_dicts(drop_rewards,finished_rewards)
        return rewards,energy_rewards
    
    
    def get_queue_lengths(self):
        queue_lengths = {}
        for server_id in self.supporting_servers:
            queue_lengths[server_id] = self.public_queues[server_id].get_queue_length()
        return queue_lengths
    