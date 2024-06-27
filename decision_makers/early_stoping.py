import os 
class EarlyStoping:
    def __init__(self,frequencies,run_folder) -> None:
        self.frequencies = frequencies
        self.step  = 0
        self.run_folder  = run_folder
        
        self.frequencies_folders = [os.path.join(run_folder,f) for f in frequencies]
        for folder in self.frequencies_folders:
            os.makedirs(folder,exist_ok=True)
            
        
        
        