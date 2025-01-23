from .decision_maker_base import DescisionMakerBase
import numpy as np


class EnergyEfficientRuleBased(DescisionMakerBase):
    """
    A rule-based decision maker that optimizes task offloading based on energy efficiency and delay.
    This class implements a decision-making strategy that chooses where to execute tasks (locally or remotely)
    by considering both energy consumption and processing delay. It evaluates different execution options
    including local processing, offloading to nearby devices, and cloud processing.
    Attributes:
        number_of_actions (int): Total number of possible actions (local + remote + cloud execution).
        local_cpu (float): Processing capacity of the local device.
        foreign_cpus (list): List of processing capacities of remote devices.
        offloading_capacities (list): Network bandwidth capacities for offloading, excluding zero values.
        private_waiting_consumption (float): Energy consumption rate while waiting in local queue.
        private_step_consumption (float): Energy consumption rate during local processing.
        offloading_waiting_consumptions (float): Energy consumption rate while waiting to offload.
        offloading_step_consumptions (float): Energy consumption rate during data transfer.
        foreign_waiting_consumption (list): Energy consumption rates while waiting in remote queues.
        foreign_step_consumption (list): Energy consumption rates during remote processing.
        delay_to_energy_ratio (float): Weight factor balancing delay vs. energy consumption (0-1).
    Methods:
        choose_action(observation, public_queues, *args, **kwargs):
            Selects the most efficient action based on current system state.
            Args:
                observation (list): Current system state including task size and waiting times.
                public_queues (list): Current load levels of remote processing queues.
            Returns:
                int: Index of the chosen action (0 for local, >0 for remote/cloud).
            The decision is made by calculating a weighted sum of delay and energy consumption
            for each possible action and selecting the one with minimum cost.
    """
    def __init__(self, 
                 number_of_actions,
                 local_cpu,
                 foreign_cpus,
                 offloading_capacities,
                 private_queue_waiting_time_consumptions,
                 private_queue_step_consumptions,
                 public_queue_waiting_time_consumptions,
                 public_queue_step_consumptions,
                 offloading_queue_waiting_time_consumptions,
                 offloading_queue_step_consumptions,
                 cloud_waiting_time_consumption,
                 cloud_step_consumption,
                 delay_to_energy_ratio,
                 *args, **kwargs):
        self.number_of_actions = number_of_actions
        self.local_cpu = local_cpu
        self.foreign_cpus = foreign_cpus
        
        self.offloading_capacities = [capacity for capacity in offloading_capacities if capacity != 0]
        
        self.private_waiting_consumption = private_queue_waiting_time_consumptions
        self.private_step_consumption = private_queue_step_consumptions
        
        
        self.offloading_waiting_consumptions = offloading_queue_waiting_time_consumptions
        self.offloading_step_consumptions = offloading_queue_step_consumptions
        
        self.foreign_waiting_consumption = [public_queue_waiting_time_consumptions] * (number_of_actions - 2)
        self.foreign_step_consumption = [public_queue_step_consumptions]* (number_of_actions - 2)
      
        self.foreign_waiting_consumption.append(cloud_waiting_time_consumption)
        self.foreign_step_consumption.append(cloud_step_consumption)
        
        self.delay_to_energy_ratio = delay_to_energy_ratio
      
    def choose_action(self, observation,public_queues ,*args, **kwargs):
        actions = np.zeros(self.number_of_actions, dtype=int)
        task_size = observation[0]
        task_computation = task_size *0.297
        local_waiting_time = observation[1]
        local_waiting_consumption = self.private_waiting_consumption * local_waiting_time
        local_procesing_time = task_computation / self.local_cpu
        local_processing_consumption = self.private_step_consumption * local_procesing_time
        
        local_time =  local_waiting_time + local_procesing_time
        local_consumption = local_waiting_consumption + local_processing_consumption
        actions[0] =  self.delay_to_energy_ratio * local_time + \
                        (1-self.delay_to_energy_ratio) * local_consumption
        
        offloading_waiting_time = observation[2]
        offloading_waiting_consumption = self.offloading_waiting_consumptions * offloading_waiting_time
        
        offloading_processing_time = task_computation / self.offloading_capacities[0]
        offloaging_processing_consumption = self.offloading_step_consumptions * offloading_processing_time
        
        for i in range(1, self.number_of_actions):
            foreign_processing_time = task_computation /(self.foreign_cpus[i-1]/public_queues[i-1])
            foreign_processing_consumption = self.foreign_step_consumption[i-1] * foreign_processing_time
            
            foreign__time = offloading_waiting_time +offloading_processing_time +foreign_processing_time
            foreign_consumption = offloading_waiting_consumption + offloaging_processing_consumption + foreign_processing_consumption
            
            
            actions[i]  = self.delay_to_energy_ratio * foreign__time + \
                        (1-self.delay_to_energy_ratio) * foreign_consumption
        return np.argmin(actions)