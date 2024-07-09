import os 
import torch
import numpy as np
class ChampionshipManager():
    def __init__(self,descision_maker, agents,windows,run_folder):
        self.descision_maker= descision_maker
        if descision_maker =='drl':
            networks = [a.Q_eval_network for a in agents]
            self.groups = self.find_groups(networks)
            self.run_folder  =run_folder
        
            self.championship_folder=  f'{run_folder}/championship'
            os.makedirs(self.championship_folder, exist_ok=True)


            self.windows = [(f,os.path.join(self.championship_folder,f'window_{f}')) for f in windows]
            for _,f in self.windows:
                os.makedirs(f, exist_ok=True)
                for i,g in enumerate(self.groups):
                    group_folder = os.path.join(f,f'group_{i}')
                    os.makedirs(group_folder, exist_ok=True)
                    score_file  = os.path.join(group_folder,'score.txt')
                    if not os.path.exists(score_file):
                        with open(score_file, 'w') as file:
                            file.write(str(-float('inf')))
       
                    
                    
       
    def find_groups(self,networks):
        identical_groups = []
        visited = set()

        for i, net1 in enumerate(networks):
            if i in visited:
                continue
            group = [i]
            for j, net2 in enumerate(networks[i+1:], start=i+1):
                if self._are_networks_identical(net1, net2):
                    group.append(j)
                    visited.add(j)
            identical_groups.append(group)
            visited.add(i)

        return identical_groups

    def _are_networks_identical(self, net1, net2):
        # Check if the architectures are the same (layer types and sequence)
        if len(list(net1.children())) != len(list(net2.children())):
            return False
        for layer1, layer2 in zip(net1.children(), net2.children()):
            if type(layer1) != type(layer2):
                return False
            # Check dimensions of the layers' parameters (weights and biases)
            if len(list(layer1.parameters())) != len(list(layer2.parameters())):
                return False
            for param1, param2 in zip(layer1.parameters(), layer2.parameters()):
                if param1.shape != param2.shape:
                    return False
        return True
    
    
    
    def step(self,rewards,agents):
        if self.descision_maker != 'drl':   
            return
        stacked_arrays = np.vstack(rewards)
        transposed_arrays = stacked_arrays.T
        for window,f_folder in self.windows:
            group_folders = os.listdir(f_folder)
            for i,g_folder in enumerate(group_folders):
                g_folder_path = os.path.join(f_folder,g_folder)
                score_file  = os.path.join(g_folder_path,'score.txt')
                with open(score_file, 'r') as file:
                    high_score = file.read().strip()
                    high_score = float(high_score)
                    best = -1
                
                for agents_in_group in self.groups[i]:
                    score  = np.mean(transposed_arrays[agents_in_group][-window:])
                    if score>high_score:
                        high_score = score
                        best = agents_in_group
                    high_score = max(high_score,score)
                    
                if best != -1:
                    with open(score_file, 'w') as file:
                        file.write(str(high_score))
                        torch.save(agents[best].Q_eval_network,os.path.join(g_folder_path,'best_model.pth'))

                                    
                

                        
    