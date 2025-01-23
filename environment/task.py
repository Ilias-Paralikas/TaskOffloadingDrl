import numpy as np
class Task():
    """A class representing a computational task in a distributed computing environment.
    This class implements a task that can be processed, transmitted, and managed across different
    servers in a distributed system. Each task has properties such as size, processing requirements,
    priority, and timing constraints.
    Attributes:
        size (float): The total size of the task in computation units
        remain (float): The remaining size of the task to be processed
        arrival_time (int): The time when the task arrives in the system
        timeout_delay (int): Maximum allowed time to process the task
        priotiry (int): Priority level of the task (higher number means higher priority)
        computational_density (float): Amount of computational resources needed per unit of task
        drop_penalty (int): Penalty score when the task is dropped/failed
        timeout_instance (int): Absolute time when the task will timeout
        origin_server_id (int): ID of the server where the task originated
        target_server_id (int): ID of the server where the task should be processed
        empty (bool): Flag indicating if the task is empty/completed
    Methods:
        drop_task(): Marks the task as dropped and returns the penalty
        finish_task(finish_time): Marks the task as finished and returns the processing time
        is_empty(): Returns whether the task is empty/completed
        process(capacity, time): Processes the task with given computational capacity
        public_process(capacity, time): Processes the task considering priority
        transmit(offloading_capacity): Transmits the task to another server
        get_size(): Returns the total size of the task
        get_relative_timeout(): Returns the timeout delay
        get_timeout(): Returns the absolute timeout instance
        get_remaining_size(): Returns the remaining size to process
        get_density(): Returns the computational density
        get_priority(): Returns the task priority
        get_target_server_id(): Returns the target server ID
        get_origin_server_id(): Returns the origin server ID
        set_origin_server_id(origin_server_id): Sets the origin server ID
        set_target_server_id(target_server_id): Sets the target server ID
        copy(): Creates and returns a copy of the task
        get_features(): Returns task features as numpy array
        get_number_of_features(): Returns the number of features
    Raises:
        AssertionError: Raised when trying to access attributes of an empty task
    """
    
    def __init__(self,
                 size :float =None,
                 arrival_time :int=0,
                 timeout_delay:int=10,
                 priotiry:int=1,
                 computational_density:float=1,
                 drop_penalty :int=1,
                 origin_server_id:int=None,
                 target_server_id:int=None,
                 ) -> None:
        if size :
            self.size = size
            self.remain = size
            self.arrival_time  = arrival_time
            self.timeout_delay = timeout_delay
            self.priotiry = priotiry
            self.computational_density = computational_density
            self.drop_penalty = drop_penalty
            self.timeout_instance = arrival_time + timeout_delay
            self.origin_server_id = origin_server_id
            self.target_server_id = target_server_id
            self.empty = False
        else:
            self.empty = True
            
    def drop_task(self) -> int:
        self.empty = True
        return self.drop_penalty
    
    
    def finish_task(self,
                    finish_time:int) ->int:
        self.empty = True
        return finish_time- self.arrival_time
    
    def is_empty(self) ->bool:
        return self.empty
    
        
    def process(self,capacity,time):
        task_processed = capacity /self.computational_density
        self.remain -= task_processed
        reward=  0
        if self.remain <=0:
            task_processed += self.remain
            reward = self.finish_task(time)
        return reward,task_processed,self.computational_density

    def public_process(self,capacity,time):
        computational_capacity =  capacity  * self.priotiry
        task_processed = computational_capacity / self.computational_density
        self.remain -= task_processed
        reward = 0
        if self.remain <=0:
            task_processed += self.remain
            reward = self.finish_task(time)
        return reward,task_processed,self.computational_density

    def transmit(self,offloading_capacity):
        transmitted_size = offloading_capacity
        self.remain -= transmitted_size
        transmitted_task =None
        
        if self.remain <=0:
            self.empty = True
            transmitted_task =  Task(size = self.size,
                                arrival_time = self.arrival_time,
                                timeout_delay = self.timeout_delay,
                                priotiry=self.priotiry,
                                computational_density = self.computational_density,
                                drop_penalty = self.drop_penalty ,
                                origin_server_id= self.origin_server_id,
                                target_server_id = self.target_server_id)
            self.empty = True
            transmitted_size += self.remain
        return transmitted_task ,transmitted_size

    def get_size(self):
        assert not self.empty
        return self.size
    def get_relative_timeout(self):
        assert not self.empty
        return self.timeout_delay
    def get_timeout(self):
        assert not self.empty
        return self.timeout_instance
    def get_remaining_size(self):
        assert not self.empty
        return self.remain
    
    def get_density(self):
        assert not self.empty
        return self.computational_density
    
    def get_priority(self):
        assert not self.empty
        return self.priotiry
    
    def get_target_server_id(self):
        assert not self.empty
        return self.target_server_id
    
    def get_origin_server_id(self):
        assert not self.empty
        return self.origin_server_id
    
    def set_origin_server_id(self,origin_server_id):
        assert not self.empty
        self.origin_server_id = origin_server_id
    
    def set_target_server_id(self,target_server_id:int)->None:
        assert not self.empty
        self.target_server_id = target_server_id
        
    def copy(self):
        return Task(size = self.size,
                    arrival_time = self.arrival_time,
                    timeout_delay = self.timeout_delay,
                    priotiry=self.priotiry,
                    computational_density = self.computational_density,
                    drop_penalty = self.drop_penalty ,
                    origin_server_id= self.origin_server_id,
                    target_server_id = self.target_server_id)
        
    def get_features(self):
        return np.array([self.size])
    def get_number_of_features(self):
        features =  self.get_features()
        return len(features)