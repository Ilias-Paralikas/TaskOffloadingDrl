import numpy as np
class Matchmaker():
    """
    A class responsible for mapping actions to server IDs in a task offloading system.
    This class maintains a mapping between numeric action indices and actual server IDs,
    allowing for translation between the action space used by the learning algorithm and 
    the actual server identifiers in the system.
    Attributes:
        id (int): Unique identifier for this Matchmaker instance
        possible_actions (numpy.ndarray): Array containing all possible server IDs that can be selected,
            including this matchmaker's own ID as the first element followed by offloading server IDs
    Methods:
        match_action(server_id, action): Maps a numeric action to its corresponding server ID
        get_rows(): Returns the array of all possible server IDs
    Args:
        id (int): The unique identifier for this Matchmaker instance
        offloading_servers (list or numpy.ndarray): List of server IDs that are available for offloading
    """
    def __init__(self,
                 id,
                 offloading_servers):
        self.id = id
        self.possible_actions = np.append(np.array([id]),offloading_servers)
    def match_action(self,server_id,action):
        assert server_id == self.id
        return self.possible_actions[action]
        
    def get_rows(self):
        return self.possible_actions