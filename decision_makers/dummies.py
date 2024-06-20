from .decision_maker_base import DescisionMakerBase
import numpy as np
class AllLocal(DescisionMakerBase):
    def __init__(self):
        super().__init__()

    def choose_action(self, state):
        return 0
    
class AllVertical(DescisionMakerBase):
    def __init__(self, number_of_actions):
        super().__init__()
        self.number_of_actions =number_of_actions

    def choose_action(self, state):
        return self.number_of_actions - 1
class AllHorizontal(DescisionMakerBase):
    def __init__(self, number_of_actions):
        super().__init__()
        self.number_of_actions =number_of_actions

    def choose_action(self, state):
        return np.random.randint(1,self.number_of_actions - 1)