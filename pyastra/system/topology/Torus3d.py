from ComplexLogicalTopology import ComplexLogicalTopology
from RingTopology import RingTopology
from Common import ComType
from BasicLogicalTopology import BasicLogicalTopology


class Torus3D(ComplexLogicalTopology):
    def __init__(self, id, total_nodes, local_dim, vertical_dim):
        horizontal_dim = total_nodes // (vertical_dim * local_dim)
        self.local_dimension = RingTopology(
            RingTopology.Dimension.Local, id, local_dim, id % local_dim, 1
        )
        self.vertical_dimension = RingTopology(
            RingTopology.Dimension.Vertical,
            id,
            vertical_dim,
            id // (local_dim * horizontal_dim),
            local_dim * horizontal_dim,
        )
        self.horizontal_dimension = RingTopology(
            RingTopology.Dimension.Horizontal,
            id,
            horizontal_dim,
            (id // local_dim) % horizontal_dim,
            local_dim,
        )

    def __del__(self):
        pass

    def get_num_of_dimensions(self):
        return 3

    def get_num_of_nodes_in_dimension(self, dimension):
        if dimension == 0:
            return self.local_dimension.get_num_of_nodes_in_dimension(0)
        elif dimension == 1:
            return self.vertical_dimension.get_num_of_nodes_in_dimension(0)
        elif dimension == 2:
            return self.horizontal_dimension.get_num_of_nodes_in_dimension(0)
        return -1

    def get_basic_topology_at_dimension(self, dimension, type):
        if dimension == 0:
            return self.local_dimension
        elif dimension == 1:
            return self.vertical_dimension
        elif dimension == 2:
            return self.horizontal_dimension
        return None