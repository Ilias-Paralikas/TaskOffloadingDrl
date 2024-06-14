import sys
import os
from .topology_generator_base import TopologyGeneratorBase

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Variabledistributor

class SkipConnections(TopologyGeneratorBase):
    def __init__(self, 
                 number_of_servers,
                 symetric,
                 skip_connections,
                 horizontal_capacities_min,
                 horizontal_capacities_max,
                 horizontal_capacities_distribution,
                 cloud_capacities_min,
                 cloud_capacities_max,
                 cloud_capacities_distribution) -> None:
        super().__init__(number_of_servers,symetric)
        self.skip_connections= skip_connections
        self.horisontal_capacity_distributor = Variabledistributor(horizontal_capacities_min,horizontal_capacities_max,horizontal_capacities_distribution)
        self.vertical_capacity_distributor = Variabledistributor(cloud_capacities_min,cloud_capacities_max,cloud_capacities_distribution)
    def create_topology(self):
        for s in range(self.number_of_servers):
            for i in range(self.skip_connections,self.number_of_servers, self.skip_connections):
                target = (s + i) % self.number_of_servers
                self.connection_matrix[s][target] = self.horisontal_capacity_distributor.generate()
            self.connection_matrix[s][-1] = self.vertical_capacity_distributor.generate()

        if self.symetric:
            self.make_symetric()
        
        return self.connection_matrix