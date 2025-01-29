import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Generate an MST using Prim's Algorithm, starting at node 0.
        """

        # Handle trivial cases of one or zero nodes.
        if self.adj_mat.size == 0:
            self.mst = self.adj_mat.copy()
            return
        if self.adj_mat.size == 1:
            self.mst = np.zeros((1,1))
            return

        self.mst = np.zeros((len(self.adj_mat),len(self.adj_mat)))
        # Use dummy node to start the while loop.
        heap = [[0, None, 0]] # Weight, v1, v2
        seen = set()
        while heap:
            weight, v1, v2 = heapq.heappop(heap)
            if v1 in seen and v2 in seen:
                # extra edge, disregard.
                continue
            if v2 in seen:
                # variable swap to prevent duplicate code.
                temp = v2
                v2 = v1
                v1 = temp
            seen.add(v2)
            self.mst[v1][v2] = weight
            self.mst[v2][v1] = weight
            # Add the newly unioned node's edges to the heap if they go to
            # nodes not already visited.
            for edge_i in range(len(self.adj_mat[v2])):
                if edge_i in seen or self.adj_mat[v2][edge_i] == 0:
                    continue
                weight = self.adj_mat[edge_i][v2]
                heapq.heappush(heap, [weight, edge_i, v2])