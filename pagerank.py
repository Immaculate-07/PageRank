import networkx as nx
from scipy.io import mmread
import time
import numpy as np
from datetime import datetime

###############
# G = nx.DiGraph()

# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
# G.add_node(4)

# G.add_edge(1,2,weight=1)
# G.add_edge(2,3,weight=1)
# G.add_edge(3,1,weight=1)
# G.add_edge(2,1,weight=1)
# G.add_edge(4,1,weight=1)


# adjacency_matrix = nx.adjacency_matrix(G)

########

## Part 1 - Desiging the Algorithm for Page Rank
#Let's create a function that runs the page rank
def pagerank(adjacency_matrix, alpha=0.85, max_iterations=100, tolerance=1e-6):
    num_nodes = adjacency_matrix.shape[0]
    #Convert to an array
    adjacency_matrix = adjacency_matrix.toarray()

    #Find the transpose of the adjacency matrix
    adjacency_matrix = np.transpose(adjacency_matrix)

    # Normalized the columns
    column_sums = np.sum(adjacency_matrix,axis=0)
    normalized_adjacency_matrix = adjacency_matrix/column_sums

    # We need to find the Transition Matrix
    Transition_Matrix = ((1-alpha)*normalized_adjacency_matrix) + (alpha/num_nodes)*np.ones((num_nodes,num_nodes))

    #Initialize the PageRank vector
    Page_rank = np.ones(num_nodes)/num_nodes

    start_time = datetime.now()

    for iterations in range(max_iterations):
        # Store the current PageRank vector
        Old_Page_rank = Page_rank.copy()

        # Calculate the new pagerake using the power method r_k+1 = Q^k*r

        Page_rank = np.dot(Transition_Matrix,Old_Page_rank)

        if np.linalg.norm(Page_rank - Old_Page_rank) < tolerance:
            break
        
        iterations = iterations+1
    end_time = datetime.now()
    running_time = end_time - start_time

    result = [Page_rank, running_time, iterations]
    return result

# result = pagerank(adjacency_matrix)
# print(result)

## Part 2 -- Reading the two sparse Graph and Converting them to an Adjacency Matrix

file_path1 = 'C:/Users/DELL/Desktop/All Python Codes/qpband.mtx'
file_path2 = 'C:/Users/DELL/Desktop/All Python Codes/bcsstk29.mtx'
matrix1 = mmread(file_path1)
matrix2 = mmread(file_path2)

G1 = nx.DiGraph(matrix1)
adjacency_matrix1 = nx.adjacency_matrix(G1)

G2 = nx.DiGraph(matrix2)
adjacency_matrix2 = nx.adjacency_matrix(G2)


# Part 3 --Implementing the Page Rank

# Sparse matrix 1
result1 = pagerank(adjacency_matrix1)
PageRank1 = result1[0]
print(PageRank1)
print(f"The number of Iterations is {result1[2]}. The time it took for the Iterations to converge is {result1[1]}")

def Pagerank_results(data):
    numeric_labels = np.array(list(range(1, len(data)+1)))

    # Get the highest and lowest values
    max_index = np.argmax(data)
    min_index = np.argmin(data)

    # Get the highest and lowest values and their corresoponding numeric labels
    max_value = data[max_index]
    min_value = data[min_index]
    max_label = numeric_labels[max_index]
    min_label = numeric_labels[min_index]

    #print the highest and lowest values along with their numberic labels
    print(f"Highest: Label {max_label}: {max_value}")
    print(f"Lowest: Label {min_label}: {min_value}")

Pagerank_results(PageRank1)

### Sparse matrix 2
result2 = pagerank(adjacency_matrix2)
PageRank2 = result2[0]
print(PageRank2)
print(f"The number of Iterations is {result1[2]}. The time it took for the Iterations to converge is {result1[1]}")

Pagerank_results(PageRank2)







