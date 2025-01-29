import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`
    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    def switch_nodes(node_a, node_b, adj_mat):
        """swap two nodes (to start algo at different node)."""
        tmp = adj_mat.copy()[:, node_a]
        adj_mat[:, node_a] = adj_mat[:, node_b]
        adj_mat[:, node_b] = tmp

        tmp = adj_mat[node_a].copy()
        adj_mat[node_a] = adj_mat[node_b]
        adj_mat[node_b] = tmp
        return adj_mat


    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    # Confirm starting at a different node yields the same overall MST value.
    for node in range(1, len(mst)):
        new_adj_mat = switch_nodes(0, node, adj_mat.copy())
        g = Graph(new_adj_mat)
        g.construct_mst()
        total = 0
        for i in range(g.mst.shape[0]):
            for j in range(i+1):
                total += g.mst[i,j]
        assert approx_equal(total, expected_weight), "MST is not deterministic"

    # Number of edges (edges in MST adj. matrix / 2) equals n - 1.
    assert np.count_nonzero(mst) // 2 == len(mst) - 1, "Incorrect # of edges"
    # Each node has at least one edge (with above, confirms no cycles).
    for row in mst:
        if len(row) > 1:
            assert np.count_nonzero(row) != 0, "MST is not fully connected"
    # Pairwise connections all less than inf --> connected MST.
    assert np.max(pairwise_distances(mst)) != np.inf, "Unconnected graph"


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """Run MST on an unconnected graph to make sure algorithm fails."""
    file_path = './data/unconnected.csv'
    g = Graph(file_path)
    g.construct_mst()
    with pytest.raises(AssertionError):
        check_mst(g.adj_mat, g.mst, 1)

def test_one_node():
    """Check that a one-node MST outputs correctly."""
    file_path = './data/one.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 0)

# Pytest yells that there's no data, but that's the point of the test.
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_nodes():
    """Check that an empty MST throws an assertion fail."""
    file_path = './data/none.csv'
    g = Graph(file_path)
    g.construct_mst()
    with pytest.raises(AssertionError):
        check_mst(g.adj_mat, g.mst, 0)