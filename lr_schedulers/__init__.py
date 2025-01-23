def constant(epochs):
    '''
    Returns 1, so the learning rate does not change.
    '''
    return 1


class Linear:
    """A linear learning rate scheduler that gradually decreases the learning rate.
    This scheduler implements a linear decay from a starting learning rate to an end learning rate
    over a specified number of epochs. The decay is achieved by subtracting a fixed step size
    from the learning rate multiplier at each epoch until reaching the target end rate.
    Parameters
    ----------
    start : float
        The initial learning rate value. Used as reference to calculate the end multiplier.
    end : float
        The final learning rate value to reach after decay. Used with start to calculate target multiplier.
    number_of_epochs : int
        The total number of epochs over which to decay the learning rate from start to end.
    Attributes
    ----------
    lr_multiplier : float
        The current learning rate multiplier value, initialized to 1.0
    lr_multpilier_end : float
        The target end multiplier calculated as end_rate/start_rate
    step : float
        The fixed amount to subtract from the multiplier each epoch to achieve linear decay
    Methods
    -------
    __call__(epochs)
        Decrements the learning rate multiplier by the step size and returns current multiplier.
        Will not decay below the target end multiplier.
    Examples
    --------
    >>> scheduler = Linear(start=0.1, end=0.01, number_of_epochs=100)
    >>> for epoch in range(100):
    >>>     current_multiplier = scheduler(epoch)  # Decays from 1.0 to 0.1 over 100 epochs
    """
    def __init__(self,start,end,number_of_epochs):
        self.lr_multiplier = 1
        self.lr_multpilier_end = end/start
        
        
        self.step = (1-self.lr_multpilier_end)/number_of_epochs
        
    def __call__(self,epochs):
        self.lr_multiplier = max(self.lr_multiplier - self.step, self.lr_multpilier_end)
        return self.lr_multiplier