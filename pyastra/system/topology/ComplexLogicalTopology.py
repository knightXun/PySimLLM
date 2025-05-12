from abc import ABC, abstractmethod

class LogicalTopology:
    class Complexity:
        Complex = "Complex"

    def __init__(self):
        self.complexity = None

class BasicLogicalTopology:
    pass

class ComType:
    pass

class ComplexLogicalTopology(LogicalTopology, ABC):
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

    