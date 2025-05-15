import math
import time
from enum import Enum

from BinaryTree import BinaryTree
from Common import ComType
from ComplexLogicalTopology import ComplexLogicalTopology
from RingTopology import RingTopology


class LocalRingGlobalBinaryTree(ComplexLogicalTopology):
    def __init__(self, id, local_dim, tree_type, total_tree_nodes, start, stride):
        self.local_dimension = RingTopology(
            RingTopology.Dimension.Local, id, local_dim, id % local_dim, 1
        )
        self.global_dimension_all_reduce = BinaryTree(
            id, tree_type, total_tree_nodes, start, stride
        )
        self.global_dimension_other = RingTopology(
            RingTopology.Dimension.Horizontal,
            id,
            total_tree_nodes,
            id // local_dim,
            local_dim,
        )

    def get_num_of_nodes_in_dimension(self, dimension):
        if dimension == 0:
            return self.local_dimension.get_num_of_nodes_in_dimension(0)
        elif dimension == 1:
            return 1
        elif dimension == 2:
            return self.global_dimension_all_reduce.get_num_of_nodes_in_dimension(0)
        else:
            return -1

    def get_num_of_dimensions(self):
        return 3

    def get_basic_topology_at_dimension(self, dimension, type):
        if dimension == 0:
            return self.local_dimension
        elif dimension == 1:
            return None
        elif dimension == 2:
            if type == ComType.All_Reduce:
                return self.global_dimension_all_reduce
            return self.global_dimension_other
        else:
            return None

    