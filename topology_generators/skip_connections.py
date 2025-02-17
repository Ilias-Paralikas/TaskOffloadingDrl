from .topology_generator_base import TopologyGeneratorBase

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Variabledistributor

class SkipConnections(TopologyGeneratorBase):
    """
    A class for generating network topologies with skip connections between servers and a cloud node.
    This class extends TopologyGeneratorBase to create specific network configurations where each server
    is connected to other servers at regular intervals (skip connections) and to a cloud node. The connections
    can be either symmetric or asymmetric, with customizable capacity distributions.
    Parameters
    ----------
    number_of_servers : int
        The total number of servers in the network topology.
    symetric : bool
        If True, creates a symmetric connection matrix where capacity(i,j) = capacity(j,i).
    skip_connections : int
        The interval at which connections are made between servers. For example, if skip_connections=2,
        each server connects to every 2nd server in the topology.
    horizontal_capacities_min : float
        Minimum capacity value for server-to-server connections.
    horizontal_capacities_max : float
        Maximum capacity value for server-to-server connections.
    horizontal_capacities_distribution : str
        The probability distribution type for generating server-to-server connection capacities.
    cloud_capacities_min : float
        Minimum capacity value for server-to-cloud connections.
    cloud_capacities_max : float
        Maximum capacity value for server-to-cloud connections.
    cloud_capacities_distribution : str
        The probability distribution type for generating server-to-cloud connection capacities.
    Attributes
    ----------
    skip_connections : int
        Stores the skip connection interval.
    horisontal_capacity_distributor : Variabledistributor
        Distributor object for generating server-to-server connection capacities.
    vertical_capacity_distributor : Variabledistributor
        Distributor object for generating server-to-cloud connection capacities.
    Methods
    -------
    create_topology()
        Generates and returns a connection matrix representing the network topology.
        The matrix includes both horizontal (server-to-server) and vertical (server-to-cloud) connections.
        Returns:
            numpy.ndarray: The connection matrix representing the network topology.
    Notes
    -----
    - The connection matrix has dimensions (number_of_servers + 1) x (number_of_servers + 1),
      where the last row/column represents the cloud node.
    - Connections wrap around the topology (modulo number_of_servers) to ensure even distribution.
    - All servers maintain a connection to the cloud node with capacities determined by the
      vertical_capacity_distributor.
    """
    def __init__(self, 
                 number_of_servers,
                 symetric,
                 skip_connections,
                 horizontal_capacities_min,
                 horizontal_capacities_max,
                 horizontal_capacities_distribution,
                 cloud_capacities_min,
                 cloud_capacities_max,
                 cloud_capacities_distribution,
                 *args, 
                 **kwargs) -> None:
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