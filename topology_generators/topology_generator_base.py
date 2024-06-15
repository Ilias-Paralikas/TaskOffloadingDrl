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
        
        