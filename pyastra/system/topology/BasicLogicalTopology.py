from enum import Enum

from LogicalTopology import LogicalTopology


class BasicLogicalTopology(LogicalTopology):
    class BasicTopology(Enum):
        Ring = 1
        BinaryTree = 2

    def __init__(self, basic_topology):
        self.basic_topology = basic_topology
        self.complexity = LogicalTopology.Complexity.Basic

    def __del__(self):
        pass

    def get_num_of_dimensions(self):
        return 1

    def get_num_of_nodes_in_dimension(self, dimension):
        raise NotImplementedError

    def get_basic_topology_at_dimension(self, dimension, type):
        return self
    