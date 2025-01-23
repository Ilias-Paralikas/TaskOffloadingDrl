from .task import Task
from utils import merge_dicts
import queue
import math
class TaskQueue():
    """
    The TaskQueue class manages a queue of tasks by overseeing the addition of new tasks,
    the retrieval of tasks for processing, the detection and handling of timeouts, and
    the computation of queue length and energy consumption.
    Attributes:
        waiting_time_consumption (float): The energy consumption rate for tasks waiting in the queue.
        step_consumption (float): The energy consumption rate multiplied by the task size per step
            while processing a task.
        current_time (int): Tracks the elapsed time steps for the queue since the last reset.
        queue_length (int): Represents the total cumulative size of all tasks waiting or being processed.
        queue (queue.Queue): A FIFO queue data structure holding tasks awaiting processing.
        current_task (Task): The task currently being processed.
    Methods:
        __init__(self, waiting_time_consumption, step_consumption) -> None:
            Initializes the queue with the specified energy consumption parameters and calls reset
            to configure the initial internal state.
        reset(self) -> None:
            Resets all internal states of the queue, including current time, queue length,
            and the task currently undergoing processing.
        add_task(self, task: Task) -> None:
            Appends a new task to the queue or assigns it as the current task if the queue is empty.
            Also updates the overarching queue length by the task's size.
        is_empty(self) -> bool:
            Determines if both the FIFO queue and the currently processed task are empty.
        current_task_is_timed_out(self) -> bool:
            Checks if the current task has exceeded its allowed timeout based on its required time.
        get_first_non_empty_element(self) -> int:
            Increments the queue's time counter by a single step, ensures the current task
            has not timed out, retrieves any necessary tasks from the queue, and returns
            any accrued rewards from tasks that were dropped due to timeouts.
        get_waiting_time(self) -> int:
            Retrieves the waiting time if applicable; otherwise, raises an error for queues
            without a waiting time feature.
        get_queue_length(self) -> int:
            Provides the cumulative size of all tasks currently in the queue or under processing.
        calculate_energy_consumption(self, task_size: int) -> float:
            Computes the energy used in a single time step, factoring in both the processing
            of the specified task (based on task size) and the waiting overhead for all remaining tasks.
    """
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
    """
    A queue implementation for processing tasks with computational capacity constraints.
    This class extends TaskQueue and manages tasks that require computational processing,
    keeping track of waiting times and processing capabilities.
    Attributes:
        computational_capacity (float): The maximum computational capacity available for processing tasks
        waiting_time (int): Current accumulated waiting time for processing tasks in the queue
    Parameters:
        computational_capacity (float): Processing power of the queue
        waiting_time_consumption (float): Energy consumption rate while tasks are waiting
        step_consumption (float): Base energy consumption per time step
    Methods:
        reset(): Resets the queue state, including waiting time
        update_waiting_time(task): Calculates and updates waiting time based on task requirements
        add_task(task): Adds a new task to the queue and updates waiting time
        step(): Processes tasks in the queue for one time step
    Returns from step():
        tuple: (rewards, energy_consumption)
            - rewards (float): Accumulated rewards from processed tasks
            - energy_consumption (float): Energy consumed during processing
    The queue processes tasks based on their computational density and size,
    ensuring tasks are completed within their timeout constraints while
    maintaining energy consumption tracking.
    """
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
    """A queue implementation for offloading tasks to remote servers.
    This class extends TaskQueue to handle task offloading to remote servers with different
    transmission capacities. It tracks waiting times and manages task transmission based on 
    server capacities.
    Attributes:
        offloading_capacities (dict): Dictionary mapping server IDs to their transmission capacities
        waiting_time (int): Current cumulative waiting time for tasks in the queue
    Parameters:
        offloading_capacities (dict): Dictionary of server transmission capacities
        waiting_time_consumption (float): Energy consumed per unit of waiting time
        step_consumption (float): Energy consumed per step of transmission
    Methods:
        reset(): Resets the queue state including waiting time
        update_waiting_time(task): Updates queue waiting time based on task size and server capacity 
        add_task(task): Adds a task to queue and updates waiting time
        step(): Executes one transmission step and returns transmitted task, reward and energy used
    Returns:
        step(): Tuple containing:
            - transmitted_task: Task object that was transmitted (or None)
            - reward: Reward value for the transmission
            - energy_consumption: Energy consumed during transmission
    Example:
        queue = OffloadingQueue(
            offloading_capacities={0: 100, 1: 200},
            waiting_time_consumption=0.1,
            step_consumption=0.5
        )
        queue.add_task(some_task)
        transmitted_task, reward, energy = queue.step()
    """
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
    """
    A class representing a public task queue in a computational offloading system.
    This class inherits from TaskQueue and implements specific processing logic for public tasks.
    It manages task processing based on available computational capacity and tracks rewards
    and energy consumption.
    Attributes:
        Inherits all attributes from TaskQueue parent class
    Methods:
        step(computational_capacity): Processes tasks in the queue based on available computational capacity
            Args:
                computational_capacity (float): The available computational resources for processing tasks
            Returns:
                tuple: A pair containing:
                    - rewards (float): The rewards obtained from processing tasks
                    - energy_consumption (float): The energy consumed during task processing
            Processing Logic:
                - If current task is empty, returns 0 rewards and energy consumption
                - Otherwise processes the current public task using the given computational capacity
                - Updates queue length based on processed tasks
                - Calculates and returns energy consumption based on computational density
    """
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
    """
    PublicQueueManager is responsible for managing multiple public queues, each associated with a
    different supporting server. It orchestrates the distribution of computational capacity among
    incoming tasks, calculates total priorities, and aggregates various rewards related to task
    processing and energy consumption.
    Attributes:
        id (Any):
            The unique identifier for this PublicQueueManager instance, corresponding to a specific server.
        computational_capacity (float):
            The total computational capacity available to be shared among tasks in the queues.
        supporting_servers (list):
            A list of server IDs that this manager oversees, each maintaining its own public queue.
        waiting_time_consumption (float):
            A factor used to penalize or account for resource usage as tasks wait in the queue over time.
        step_consumption (float):
            A factor representing resource or energy consumption applied at each step of queue processing.
        public_queues (dict):
            A dictionary mapping each server ID to its corresponding PublicQueue instance.
    Methods:
        __init__(id, computational_capacity, supporting_servers, waiting_time_consumption, step_consumption):
            Initializes the PublicQueueManager with the given configuration. Instantiates a PublicQueue
            for each supporting server, setting up all necessary consumption parameters.
        reset():
            Resets all existing public queues, clearing any stored tasks and reverting each queue to its
            initial state.
        get_public_queue_server_length(server_id):
            Returns the current number of tasks for the specified server's public queue.
        get_priorities():
            Calculates the total priority across all non-empty queues. Priority is derived from the tasks
            currently being processed in those queues.
        get_active_queues():
            Counts how many queues currently contain tasks. This helps determine how many queues are actively
            in use.
        add_tasks(recieved_tasks):
            Adds a list of new tasks to their appropriate public queue, ensuring each task is placed in the
            correct server queue based on its origin.
        step():
            Performs a single simulation step for all managed queues by distributing computational capacity
            according to task priorities. Collects information about tasks that finish processing or are
            dropped, as well as associated energy consumption, returning these values for further analysis.
        get_queue_lengths():
            Retrieves the number of tasks queued in each server's public queue, returning a dictionary keyed
            by server ID.
    """
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
    