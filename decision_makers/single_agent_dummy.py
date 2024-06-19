from .decision_maker_base import DescisionMakerBase


class SingleAgentDummy(DescisionMakerBase):
    def __init__( self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def choose_action(self,observation, *args, **kwargs):
        local_waiting_time=  observation[-2]
        transmision_time = observation[-1] 
        if local_waiting_time < transmision_time:
            return 0
        else:    
            return 1