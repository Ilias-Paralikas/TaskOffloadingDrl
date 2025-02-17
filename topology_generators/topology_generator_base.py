import numpy as np
import matplotlib.pyplot as plt

class TopologyGeneratorBase():
    def __init__(self,
                 number_of_servers,
                 symetric,       
                 *args, 
                 **kwargs) -> None:
        self.number_of_servers= number_of_servers
        self.symetric = symetric
        self.number_of_clouds= 1
        self.connection_matrix= np.zeros((self.number_of_servers,self.number_of_servers+1))
    def create_topology(self,*args, **kwargs):
        pass
    def make_symetric(self):
        for i in range(self.number_of_servers):
            for j in range(i + 1, self.number_of_servers):
                self.connection_matrix[j][i] = self.connection_matrix[i][j]
                

def plot_matrix(connection_matrix,path):
    def plot_matrix(connection_matrix, path):
        """
        Plots and saves a heatmap visualization of a connection matrix.
        This function creates a visual representation of a connection matrix using a heatmap,
        where each cell's value is represented by both color intensity and numeric annotation.
        Non-zero values are displayed as text within their respective cells.
        Args:
            connection_matrix (array-like): A 2D matrix/array representing connection values 
                between nodes. Can be a list of lists or numpy array.
            path (str): File path where the resulting plot should be saved.
        Features:
            - Automatically scales font size based on matrix dimensions
            - Uses 'viridis' colormap for visualization
            - Includes a colorbar for reference
            - Only annotates non-zero values for clarity
            - Places x-axis labels at the top of the plot
        Returns:
            None. The plot is saved to the specified path.
        Example:
            >>> matrix = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
            >>> plot_matrix(matrix, "output_plot.png")
        """
    connection_matrix = np.array(connection_matrix)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(connection_matrix, cmap='viridis')
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
    
    plt.savefig(path)
        
        