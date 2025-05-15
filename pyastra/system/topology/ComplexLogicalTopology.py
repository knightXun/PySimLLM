from abc import ABC, abstractmethod

from LogicalTopology import LogicalTopology
from BasicLogicalTopology import BasicLogicalTopology
from Common import ComType

class ComplexLogicalTopology(LogicalTopology):
    def __init__(self):
        super().__init__()
        self.complexity = LogicalTopology.Complexity.Complex

    @abstractmethod
    def get_num_of_nodes_in_dimension(self, dimension):
        pass

    @abstractmethod
    def get_num_of_dimensions(self):
        pass

    @abstractmethod
    def get_basic_topology_at_dimension(self, dimension, type):
        pass

    