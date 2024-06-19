import numpy as np

def find_maxes(arrays):
    # Assuming all arrays have the same shape and at least one array is present
    max_array = np.copy(arrays[0])  # Initialize max_array with the first array's values
    
    for array in arrays[1:]:  # Start from the second array
        np.maximum(max_array, array, out=max_array)  # Compare and store the max values
    
    return max_array

# Example usage
arrays = [
    np.array([1, 2 ,3, 4]),
    np.array([2, 1,100, 0]),
    np.array([0, 3,1, 5])
]

max_array = find_maxes(arrays)
print(max_array)