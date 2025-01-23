
class DescisionMakerBase():
  """
  DescisionMakerBase is a foundational class designed to provide a consistent interface 
  for decision-making in task offloading or scheduling contexts. 
  This abstract-like class is meant to be extended by more specialized decision makers 
  that implement deep reinforcement learning or other algorithmic strategies. It 
  provides hooks for model storage, loading, and decision logic, ensuring common 
  functionalities are centralized.
  Attributes:
    None
  Args:
    *args: 
      Variable-length argument list; can be used to pass custom parameters 
      to child classes.
    **kwargs: 
      Arbitrary keyword arguments; can be used to pass configuration details 
      or hyperparameters to child classes.
  Methods:
    store_model(*args, **kwargs):
      Persists the current model or policy representation to storage 
      for later retrieval or analysis.
    load_model(*args, **kwargs):
      Retrieves a previously saved model or policy representation, 
      restoring the decision-maker's state.
    store_transitions(*args, **kwargs):
      Records state transitions or experience tuples needed for 
      training in reinforcement learning or other algorithms.
    choose_action(*args, **kwargs):
      Decides which action to take based on the current state or 
      observation, returning the chosen action.
    learn(*args, **kwargs):
      Implements the training routine, refining the model or policy 
      using stored experiences and performance feedback.
    get_epsilon(*args, **kwargs):
      Returns the current exploration rate (if any) for exploration-exploitation 
      trade-offs. By default, it indicates a disabled or invalid value.
    get_learning_rate(*args, **kwargs):
      Returns the learning rate (if any) for gradient-based methods. 
      By default, it indicates a disabled or invalid value.
    store_champion(*args, **kwargs):
      Saves the best-performing model or policy encountered so far, 
      allowing easy reversion or further inspection.
    average_Weights(*args, **kwargs):
      Merges or averages the weights of multiple models, possibly to 
      stabilize training or incorporate knowledge from different 
      training phases.
    reset_lstm_history(*args, **kwargs):
      Clears any hidden states or stored sequences in recurrent models, 
      allowing fresh starts or resets in sequence-based decision methods.
  """
  

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