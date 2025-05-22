import enum

class LogicalTopology:
    
    class Complexity(enum.Enum):
        Basic = "Basic"
        Complex = "Complex"

    def __init__(self):
        self.complexity = None

    def get_topology(self):
        return self

    @staticmethod
    def get_reminder(number, divisible):
        if number >= 0:
            return number % divisible
        else:
            return (number + divisible) % divisible

    def __del__(self):
        pass

    def get_num_of_dimensions(self):
        raise NotImplementedError

    def get_num_of_nodes_in_dimension(self, dimension):
        raise NotImplementedError

    def get_basic_topology_at_dimension(self, dimension, type):
        raise NotImplementedError