import numpy as np
import matplotlib.pyplot as plt

def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries by combining their values for matching keys and preserving unique keys.
    For shared keys, values are added together.
    
    Args:
        dict1 (dict): First dictionary to merge
        dict2 (dict): Second dictionary to merge
    
    Returns:
        dict: A new dictionary containing combined values from both input dictionaries
    """
    result = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        if key in dict1 and key in dict2:
            result[key] = dict1[key] + dict2[key]
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]
    return result

def dict_to_array(dict, length):
    """
    Converts a dictionary to a numpy array where dictionary keys represent indices
    and values are placed at those indices. Non-specified indices are filled with zeros.
    
    Args:
        dict (dict): Input dictionary with integer keys
        length (int): Length of the output array
    
    Returns:
        np.ndarray: A numpy array of specified length with dictionary values at corresponding indices
    """
    array = np.zeros(length, dtype=np.float32)
    for key, value in dict.items():
        array[key] = value
    return array

def remove_diagonal_and_reshape(matrix):
    """
    Creates a new matrix where each row contains all elements from the original matrix
    except the diagonal element for that row.
    
    Args:
        matrix (np.ndarray): Square input matrix
    
    Returns:
        np.ndarray: Reshaped matrix with diagonal elements removed from each row
    """
    n = matrix.shape[0]
    new_matrix = np.empty((n, n))
    for i in range(n):
        new_matrix[i] = np.concatenate([matrix[i, :i], matrix[i, i+1:]])
    return new_matrix

class Variabledistributor:
    """
    A class that generates random values according to specified distribution patterns.
    Supports uniform distribution, random choice from range, and constant value generation.
    
    Args:
        min (number): Minimum value for the distribution range
        max (number): Maximum value for the distribution range
        distribution (str): Type of distribution ('uniform', 'choice', or 'constant')
    """
    def __init__(self, min, max, distribution):
        self.min = min
        self.max = max
        self.distribution_choices = {
            "uniform": lambda: np.random.uniform(self.min, self.max),
            "choice": lambda: np.random.choice(range(self.min, self.max+1)),
            'constant': lambda: self.max,
        }
        self.distribution = self.distribution_choices[distribution]
    
    def generate(self):
        """
        Generates a random value according to the specified distribution.
        
        Returns:
            number: A random value based on the configured distribution
        """
        return self.distribution()

def visualize_2d_array(connection_matrix, save_location, cmap='viridis'):
    """
    Creates and saves a heatmap visualization of a 2D array with numerical annotations.
    Non-zero values are annotated in the cells.
    
    Args:
        connection_matrix (np.ndarray): 2D array to visualize
        save_location (str): File path where the visualization should be saved
        cmap (str, optional): Colormap to use for the heatmap. Defaults to 'viridis'
    """
    fig, ax = plt.subplots()
    heatmap = ax.imshow(connection_matrix, cmap=cmap)
    plt.colorbar(heatmap)
    ax.set_title('Connection Matrix')
    num_rows, num_cols = connection_matrix.shape
    font_size = min(fig.get_size_inches()) * 72 / max(num_rows, num_cols) * 0.4
    
    for i in range(num_rows):
        for j in range(num_cols):
            if connection_matrix[i, j] != 0:
                text = ax.text(j, i, int(connection_matrix[i, j]),
                             ha="center", va="center", color="w",
                             fontsize=font_size)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.savefig(save_location)

def sum_dicts_in_positions(list_of_lists):
    """
    Sums dictionaries that are in the same positions across multiple sublists.
    For each position, creates a new dictionary with combined values from all sublists.
    
    Args:
        list_of_lists (list): A list containing sublists of dictionaries
    
    Returns:
        list: A list of dictionaries where each dictionary contains the sum of values
              from corresponding positions in the input sublists
    """
    num_dicts = len(list_of_lists[0])   
    summed_dicts = []

    for i in range(num_dicts):
        sum_dict = {}
        for sublist in list_of_lists:
            for key, value in sublist[i].items():
                if key in sum_dict:
                    sum_dict[key] += value
                else:
                    sum_dict[key] = value
        summed_dicts.append(sum_dict)

    return summed_dicts
