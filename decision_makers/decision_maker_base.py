class DescisionMakerBase():
  def __init__( self, *args, **kwargs):
    pass
  def store_model(self, *args, **kwargs):
    pass    
  def load_model(self, *args, **kwargs):
    pass  
  def store_transitions(self, *args, **kwargs):
    pass
  def choose_action(self, *args, **kwargs):
   pass
  def learn(self, *args, **kwargs):
    pass
  def get_epsilon(self, *args, **kwargs):
    return -1
  def get_learning_rate(self, *args, **kwargs):
    return -1
  def store_champion(self, *args, **kwargs):
    pass
  def average_Weights(self,*args, **kwargs):
    pass
  def reset_lstm_history(self,*args, **kwargs):
    pass