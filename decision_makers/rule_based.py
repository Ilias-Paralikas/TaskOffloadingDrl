from .decision_maker_base import DescisionMakerBase
import numpy as np


class RuleBased(DescisionMakerBase):
    def __init__(self, number_of_actions,local_cpu,foreign_cpus, *args, **kwargs):
        self.number_of_actions = number_of_actions
        self.local_cpu = local_cpu
        self.foreign_cpus = foreign_cpus


    def choose_action(self, observation,public_queues ,*args, **kwargs):
        actions = np.zeros(self.number_of_actions, dtype=int)
        task_size = observation[0]
        task_size = task_size *0.297
        local_waiting_time = observation[1]
        local_procesing_time = task_size / self.local_cpu
        actions[0] = local_waiting_time + local_procesing_time
        
        
        offloading_waiting_time = observation[2]
        for i in range(1, self.number_of_actions):
            foreign_processing_time = task_size /(self.foreign_cpus[i-1]/public_queues[i-1])
            actions[i]  = offloading_waiting_time + foreign_processing_time
        return np.argmin(actions)