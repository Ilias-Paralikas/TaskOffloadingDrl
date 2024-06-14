import numpy as np
class Matchmaker():
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