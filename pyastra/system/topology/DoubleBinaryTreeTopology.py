import math

from ComplexLogicalTopology import ComplexLogicalTopology
from LocalRingGlobalBinaryTree import BinaryTree


class DoubleBinaryTreeTopology(ComplexLogicalTopology):
    def __init__(self, id, total_tree_nodes, start, stride):
        if id == 0:
            print(f"Node 0: Double binary tree created with total nodes: {total_tree_nodes} ,start: {start} ,stride: {stride}")
        self.DBMAX = BinaryTree(id, BinaryTree.TreeType.RootMax, total_tree_nodes, start, stride)
        self.DBMIN = BinaryTree(id, BinaryTree.TreeType.RootMin, total_tree_nodes, start, stride)
        self.counter = 0

    def __del__(self):
        # del self.DBMAX
        # del self.DBMIN
        return

    def get_topology(self):
        if self.counter % 2 == 0:
            ans = self.DBMAX
        else:
            ans = self.DBMIN
        self.counter += 1
        return ans

    def get_num_of_dimensions(self):
        return 1

    def get_num_of_nodes_in_dimension(self, dimension):
        return self.DBMIN.get_num_of_nodes_in_dimension(0)

    def get_basic_topology_at_dimension(self, dimension, type):
        if dimension == 0:
            return self.get_topology().get_basic_topology_at_dimension(0, type)
        else:
            return None