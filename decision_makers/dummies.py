from .decision_maker_base import DescisionMakerBase
import numpy as np
class AllLocal(DescisionMakerBase):
    def choose_action(self,*args, **kwargs):
        return 0
    
class AllVertical(DescisionMakerBase):
    def __init__(self, number_of_actions,*args, **kwargs):
        self.number_of_actions =number_of_actions

    def choose_action(self,*args, **kwargs):
        return self.number_of_actions - 1
    
    
class AllHorizontal(DescisionMakerBase):
    def __init__(self, number_of_actions,*args, **kwargs):
        super().__init__()
        self.number_of_actions =number_of_actions

    def choose_action(self, *args, **kwargs):
        return np.random.randint(1,self.number_of_actions - 1)
    
class Random(DescisionMakerBase):
    def __init__(self, number_of_actions,*args, **kwargs):
        self.number_of_actions =number_of_actions
    def choose_action(self, *args, **kwargs):
        return np.random.randint(0,self.number_of_actions )
    