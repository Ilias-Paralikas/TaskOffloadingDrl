import numpy as np
import matplotlib.pyplot as plt

def merge_dicts(dict1, dict2):
    # Initialize the result dictionary
    result = {}
    # Combine keys from both dictionaries
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        # If key is in both dictionaries, add their values
        if key in dict1 and key in dict2:
            result[key] = dict1[key] + dict2[key]
        # If key is only in one of the dictionaries, preserve its value
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]
    return result

def dict_to_array(dict, length):
    # Initialize an array of zeros with the given length
    array = np.zeros(length, dtype=np.float32)
    # Iterate through the dictionary
    for key, value in dict.items():
        # If the key is within the bounds of the array length, set the corresponding value
        if 0 <= key < length:
            array[key] = value
    return array

def remove_diagonal_and_reshape(matrix):
    n = matrix.shape[0]
    new_matrix = np.empty((n, n))
    for i in range(n):
        new_matrix[i] = np.concatenate([matrix[i, :i], matrix[i, i+1:]])
    return new_matrix

class Variabledistributor():
    def __init__(self,min,max,distribution):
        self.min = min
        self.max = max
        self.distribution_choices = {
            "uniform":lambda :np.random.uniform(self.min,self.max),
            "choice": lambda :np.random.choice(range(self.min,self.max+1)),
            'constant':lambda :self.max,
        }
        self.distribution = self.distribution_choices[distribution]

    
    def generate(self):
        return self.distribution()
    
    

def visualize_2d_array(connection_matrix,save_location, cmap='viridis'):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(connection_matrix, cmap=cmap)
    plt.colorbar(heatmap)
    ax.set_title('Connection Matrix')
    num_rows, num_cols = connection_matrix.shape
    font_size = min(fig.get_size_inches()) * 72 / max(num_rows, num_cols) * 0.4
    
    # Annotate each cell with the numeric value
    for i in range(num_rows):
        for j in range(num_cols):
            if connection_matrix[i, j] != 0:  # Only annotate non-zero values
                text = ax.text(j, i, int(connection_matrix[i, j]),
                               ha="center", va="center", color="w",
                               fontsize=font_size)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.savefig(save_location)
    
def sum_dicts_in_positions(list_of_lists):
    # Assuming all sublists have the same length
    num_dicts = len(list_of_lists[0])   
    summed_dicts = []

    for i in range(num_dicts):
        # Initialize a new dictionary to store the sums for the current position
        sum_dict = {}
        for sublist in list_of_lists:
            for key, value in sublist[i].items():
                if key in sum_dict:
                    sum_dict[key] += value
                else:
                    sum_dict[key] = value
        summed_dicts.append(sum_dict)

    return summed_dicts
