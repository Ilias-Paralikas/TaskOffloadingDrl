def constant(epochs):
    return 1


class Linear:
    def __init__(self,start,end,number_of_epochs):
        self.lr_multiplier = 1
        self.lr_multpilier_end = end/start
        
        
        self.step = (1-self.lr_multpilier_end)/number_of_epochs
        
    def __call__(self,epochs):
        print(self.lr_multiplier)
        print(self.step)
        print(self.lr_multpilier_end)
        self.lr_multiplier = max(self.lr_multiplier - self.step, self.lr_multpilier_end)
        return self.lr_multiplier