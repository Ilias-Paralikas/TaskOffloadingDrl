from .task import Task
from utils import Variabledistributor
import numpy as np


class TaskGenerator():
    def __init__(self,
                id,
                episode_time,
                task_arrive_probability,
                size_min,
                size_max,
                size_distribution,
                timeout_delay_min,
                timeout_delay_max,
                timeout_delay_distribution,
                priotiry_min,
                priotiry_max,
                priotiry_distribution,
                computational_density_min,
                computational_density_max,
                computational_density_distribution,
                drop_penalty_min,   
                drop_penalty_max,
                drop_penalty_distribution,):
        self.id = id
        self.episode_time= episode_time
        self.task_arrive_probability = task_arrive_probability
        self.size_distributor =Variabledistributor(size_min,size_max,size_distribution)
        self.timeout_distributor =Variabledistributor(timeout_delay_min,timeout_delay_max,timeout_delay_distribution)
        self.priotiry_distributor =Variabledistributor(priotiry_min,priotiry_max,priotiry_distribution)
        self.computational_density_distributor =Variabledistributor(computational_density_min,computational_density_max,computational_density_distribution)
        self.drop_penalty_distributor =Variabledistributor(drop_penalty_min,drop_penalty_max,drop_penalty_distribution)
        self.current_time = 0
        self.size_max  = size_max
        self.timeout_delay_max = timeout_delay_max
        self.priotiry_max = priotiry_max
        self.computational_density_max = computational_density_max
        self.drop_penalty_max = drop_penalty_max
        
    def reset(self):
        self.current_time = -1
    def step(self):
        self.current_time +=1
        if self.current_time < self.episode_time:
            if np.random.rand() > self.task_arrive_probability:
                return None
            return self.generate()
        return None
   
    def generate(self):
        size = self.size_distributor.generate()
        timeout_delay = self.timeout_distributor.generate()
        priotiry = self.priotiry_distributor.generate()
        computational_density = self.computational_density_distributor.generate()
        drop_penalty = self.drop_penalty_distributor.generate()
        return Task(size=size,
                    arrival_time=self.current_time,
                    timeout_delay=timeout_delay,
                    priotiry=priotiry,
                    computational_density=computational_density,
                    drop_penalty=drop_penalty)
    def get_number_of_features(self):
       return self.generate().get_number_of_features()
   
    def get_maxs(self):
            return np.array([self.size_max])       