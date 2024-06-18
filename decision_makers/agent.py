import copy
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np 

from collections import deque
import os 
from .decision_maker_base import DescisionMakerBase



# Set seeds for reproducibility
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Ensure CUDA operations are deterministic
# Note: This may impact performance
torch.backends.cudnn.deterministic = True

class DeepQNetwork(nn.Module):
    def __init__(self,
                 state_dimensions,
                 lstm_input_shape,
                 lstm_output_shape,
                 number_of_actions,
                 hidden_layers,
                 lstm_layers,
                 dueling,
                 dropout_rate):
        super(DeepQNetwork,self).__init__()

        self.state_dimensions = state_dimensions
        self.lstm_input_shape =lstm_input_shape
        self.lstm_output_shape = lstm_output_shape
        self.number_of_actions = number_of_actions
        self.hidden_layers = hidden_layers
        self.lstm_layers = lstm_layers
        self.dueling = dueling
        
        self.lstm_unit = nn.LSTM(input_size= lstm_input_shape,hidden_size =lstm_output_shape,num_layers =lstm_layers,batch_first=True)

        layers = []
        last_layer_size = state_dimensions+lstm_output_shape
        for next_layer_size in hidden_layers:
            layers.append(nn.Linear(last_layer_size, next_layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_layer_size = next_layer_size

        self.sequential  = nn.Sequential(*layers)
        
        if self.dueling:
            self.value_layer = nn.Linear(last_layer_size, 1)
            self.advantage_layer = nn.Linear(last_layer_size, self.number_of_actions)
        else:
            self.output_layer = nn.Linear(last_layer_size, self.number_of_actions)
        
    def forward(self,state,lstm_input):
        batch_size = lstm_input.shape[0]
        h0 = torch.zeros(self.lstm_layers,batch_size,self.lstm_output_shape).to(lstm_input.device)
        c0 = torch.zeros(self.lstm_layers,batch_size,self.lstm_output_shape).to(lstm_input.device)
        lstm_output,_ = self.lstm_unit(lstm_input,[h0,c0])
        lstm_output = lstm_output[:,-1]


        combined_input = torch.cat((state, lstm_output), dim=1)
        sequential_output = self.sequential(combined_input)
        
        if self.dueling:
            value = self.value_layer(sequential_output)
            advantage = self.advantage_layer(sequential_output)
            q_values = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        else:
            q_values = self.output_layer(sequential_output)
            
        q_values = nn.Sigmoid()(q_values)
        return q_values
    

class Agent(DescisionMakerBase):
    def __init__(   self,
                    id,
                    state_dimensions,
                    lstm_shape,
                    number_of_actions,
                    hidden_layers,
                    lstm_layers,
                    lstm_time_step,
                    dropout_rate,
                    device,
                    epsilon,
                    epsilon_decrement,
                    epsilon_end,
                    gamma,
                    learning_rate,
                    loss_function,
                    optimizer,
                    checkpoint_folder,
                    save_model_frequency,
                    update_weight_percentage,
                    memory_size,
                    batch_size,
                    replace_target_iter,
                    dueling= True,):
        self.id = id
        self.state_dimensions = state_dimensions
        self.lstm_shape = lstm_shape
        self.number_of_actions = number_of_actions
        self.hidden_layers = hidden_layers
        self.lstm_layers = lstm_layers
        self.device = device
    
        self.lstm_time_step =lstm_time_step
        self.lstm_history = deque(maxlen=self.lstm_time_step)
        for _ in range(self.lstm_time_step):
            self.lstm_history.append(np.zeros([self.lstm_shape]))

        
        self.epsilon = epsilon
        self.epsilon_decrement = epsilon_decrement
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.learning_rate = learning_rate
    
        self.checkpoint_folder = checkpoint_folder + '/agent_'+str(id)+'.pth'
        self.save_model_frequency =save_model_frequency
        self.update_weight_percentage = update_weight_percentage
        
        
        self.Q_eval_network = DeepQNetwork(state_dimensions = state_dimensions,
                                             lstm_input_shape = lstm_shape,
                                             lstm_output_shape = lstm_shape,
                                             number_of_actions = number_of_actions,
                                             hidden_layers = hidden_layers,
                                             lstm_layers = lstm_layers,
                                             dropout_rate = dropout_rate,
                                                dueling = dueling).to(device)
        self.load_model()
        self.Q_target_network  = copy.deepcopy(self.Q_eval_network).to(device)
        
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.Q_eval_network.parameters(),lr=learning_rate)
            
        self.memory_counter= 0
        self.learn_step_counter = 0
        self.replace_target_iter=  replace_target_iter
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.state_memory = np.zeros((self.memory_size, state_dimensions),dtype=np.float32)
        self.lstm_memory =np.zeros((self.memory_size, self.lstm_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, state_dimensions),dtype=np.float32)
        self.new_lstm_memory =np.zeros((self.memory_size, self.lstm_shape),dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size,dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size,dtype=np.int64)
        self.terminal_memory= np.zeros(self.memory_size,dtype=bool)

    def store_transitions(self,state,lstm_state,action,reward,new_state,new_lstm_state,done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] =state
        self.lstm_memory[index]  =lstm_state
        self.action_memory[index] =action
        self.reward_memory[index] =reward
        self.new_state_memory[index] = new_state
        self.new_lstm_memory[index] = new_lstm_state
        self.terminal_memory[index] = done

        self.memory_counter +=1
        
    def reset_lstm_history(self):
        self.lstm_history = deque(maxlen=self.lstm_time_step)
        for _ in range(self.lstm_time_step):
            self.lstm_history.append(np.zeros([self.lstm_shape]))
    
    def choose_action(self,observation,lstm_state):     
        self.lstm_history.append(lstm_state)
        with torch.no_grad():
            if np.random.uniform() > self.epsilon:
                observation_np = np.expand_dims(observation,axis=0) # Convert the list of NumPy arrays to a single NumPy array
                lstm_history_np = np.expand_dims(self.lstm_history,axis=0)  # Convert the list of NumPy arrays to a single NumPy array

                observation = torch.tensor(observation_np,dtype=torch.float32).to(self.device)
                lstm_input = torch.tensor(lstm_history_np,dtype=torch.float32).to(self.device)
                action = np.argmax(self.Q_eval_network(observation, lstm_input).detach().cpu().numpy())
            else:
                action = np.random.choice(self.number_of_actions)
        return action
    
    
    
    
    def load_model(self,checkpoint_folder=None):
        if not checkpoint_folder:
            checkpoint_folder = self.checkpoint_folder
        try:
            if os.path.isfile(checkpoint_folder):
                self.Q_eval_network = torch.load(checkpoint_folder,map_location=self.device)
                print('model weights loaded')
            else:
                print('weights folder not found')
        except Exception as e:
            print('An error occurred while loading the model weights:', str(e)) 
    
    def store_model(self,path=None):
        if not path:
            path = self.checkpoint_folder
        torch.save(self.Q_eval_network, path)
        
    def learn(self):
        def weighted_add_state_dicts():
            eval_state_dict = self.Q_eval_network.state_dict()
            target_state_dict = self.Q_target_network.state_dict()
            new_state_dict = {}
            for key in target_state_dict.keys():
                new_state_dict[key] = self.update_weight_percentage*eval_state_dict[key]  + target_state_dict[key] * (1 - self.update_weight_percentage)
            return new_state_dict
        def get_lstm_sequence(index):
            start_index = max(0, index - self.lstm_time_step + 1)
            return torch.tensor(self.lstm_memory[start_index:index])

        
    
        if self.memory_counter <= self.batch_size+self.lstm_time_step:
            return 

        self.Q_eval_network.train()

        self.learn_step_counter +=1
        if self.learn_step_counter % self.replace_target_iter == 0:
            new_target_weights =  weighted_add_state_dicts()
            self.Q_target_network.load_state_dict(new_target_weights)

        self.optimizer.zero_grad()
        



        max_memory = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(range(self.lstm_time_step,max_memory), self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_memory[batch_indices]).to(self.device)
        lstm_sequence_batch = [get_lstm_sequence(index) for index in batch_indices]
        lstm_sequence_batch = torch.stack(lstm_sequence_batch).to(self.device)
        action_batch = torch.tensor(self.action_memory[batch_indices]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch_indices]).to(self.device)
        next_state_batch = torch.tensor(self.new_state_memory[batch_indices]).to(self.device)
        next_lstm_sequence_batch = [get_lstm_sequence(index + 1) for index in batch_indices]  
        next_lstm_sequence_batch = torch.stack(next_lstm_sequence_batch).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch_indices]).to(self.device)


        q_eval = self.Q_eval_network(state_batch, lstm_sequence_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_next_eval = self.Q_eval_network(next_state_batch, next_lstm_sequence_batch)
        next_actions = torch.argmax(q_next_eval, dim=1)
        q_next_target = self.Q_target_network(next_state_batch, next_lstm_sequence_batch)
        q_target_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        mask = (terminal_batch == 0)
        q_target_next = q_target_next * mask
        q_target = reward_batch + self.gamma * q_target_next

        loss = self.loss_function(q_eval,q_target)
        # print("Old loss :",loss.item())
        loss.backward()
        self.optimizer.step()
        
        # x = self.Q_eval_network(state_batch, lstm_sequence_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        # print("New Loss :",self.loss_function(x,q_target).item())
        self.epsilon = max(self.epsilon - self.epsilon_decrement, self.epsilon_end)

        if self.learn_step_counter % self.save_model_frequency == 0 :
            self.store_model()
            
    def get_epsilon(self):
        return self.epsilon