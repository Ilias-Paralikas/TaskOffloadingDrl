from .topology_generator_base import TopologyGeneratorBase

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Variabledistributor

class FullyConnected(TopologyGeneratorBase):
    """Generates a fully connected topology for a server network with cloud connections.
    This class creates a topology where each server is connected to every other server
    and to a cloud node. The connections can be either symmetric or asymmetric, with
    configurable capacity distributions for both horizontal (server-to-server) and
    vertical (server-to-cloud) connections.
    Parameters
    ----------
    number_of_servers : int
        The total number of servers in the network.
    symetric : bool
        If True, the connection matrix will be symmetric (i.e., capacity from A to B equals B to A).
    horizontal_capacities_min : float
        Minimum capacity value for server-to-server connections.
    horizontal_capacities_max : float
        Maximum capacity value for server-to-server connections.
    horizontal_capacities_distribution : str
        Distribution type for generating server-to-server connection capacities.
    cloud_capacities_min : float
        Minimum capacity value for server-to-cloud connections.
    cloud_capacities_max : float
        Maximum capacity value for server-to-cloud connections.
    cloud_capacities_distribution : str
        Distribution type for generating server-to-cloud connection capacities.
    Attributes
    ----------
    horisontal_capacity_distributor : Variabledistributor
        Generates capacity values for server-to-server connections.
    vertical_capacity_distributor : Variabledistributor
        Generates capacity values for server-to-cloud connections.
    connection_matrix : numpy.ndarray
        Matrix representing the network topology where entry [i][j] represents
        the connection capacity from server i to server j. The last column
        represents connections to the cloud.
    Methods
    -------
    create_topology()
        Generates and returns the connection matrix representing the fully connected topology.
        Returns:
            numpy.ndarray: The generated connection matrix including cloud connections.
    """
    def __init__(self, 
                 number_of_servers,
                 symetric,
                 horizontal_capacities_min,
                 horizontal_capacities_max,
                 horizontal_capacities_distribution,
                 cloud_capacities_min,
                 cloud_capacities_max,
                 cloud_capacities_distribution,
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(number_of_servers,symetric)
        self.horisontal_capacity_distributor = Variabledistributor(horizontal_capacities_min,horizontal_capacities_max,horizontal_capacities_distribution)
        self.vertical_capacity_distributor = Variabledistributor(cloud_capacities_min,cloud_capacities_max,cloud_capacities_distribution)
    def create_topology(self):
        for s in range(self.number_of_servers):
            for i in range(self.number_of_servers):
                target = (s + i) % self.number_of_servers
                if target != s:
                    self.connection_matrix[s][target] = self.horisontal_capacity_distributor.generate()
            self.connection_matrix[s][-1] = self.vertical_capacity_distributor.generate()

        if self.symetric:
            self.make_symetric()
        
        return self.connection_matrix