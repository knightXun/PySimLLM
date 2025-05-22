from ComplexLogicalTopology import ComplexLogicalTopology
from Common import ComType
from DoubleBinaryTreeTopology import DoubleBinaryTreeTopology
from RingTopology import RingTopology


class LocalRingNodeA2AGlobalDBT(ComplexLogicalTopology):
    def __init__(self, id, local_dim, node_dim, total_tree_nodes, start, stride):
        self.global_dimension_all_reduce = DoubleBinaryTreeTopology(id, total_tree_nodes, start, stride)
        self.global_dimension_other = RingTopology(RingTopology.Dimension.Vertical, id, total_tree_nodes, id // (local_dim * node_dim), local_dim * node_dim)
        self.local_dimension = RingTopology(RingTopology.Dimension.Local, id, local_dim, id % local_dim, 1)
        self.node_dimension = RingTopology(RingTopology.Dimension.Horizontal, id, node_dim, (id % (local_dim * node_dim)) // local_dim, local_dim)

    def __del__(self):
        return

    def get_basic_topology_at_dimension(self, dimension, type):
        if dimension == 0:
            return self.local_dimension
        elif dimension == 1:
            return self.node_dimension
        elif dimension == 2:
            if type == ComType.All_Reduce:
                return self.global_dimension_all_reduce.get_basic_topology_at_dimension(2, type)
            else:
                return self.global_dimension_other
        return None

    def get_num_of_nodes_in_dimension(self, dimension):
        if dimension == 0:
            return self.local_dimension.get_num_of_nodes_in_dimension(0)
        elif dimension == 1:
            return self.node_dimension.get_num_of_nodes_in_dimension(0)
        elif dimension == 2:
            return self.global_dimension_other.get_num_of_nodes_in_dimension(0)
        return -1

    def get_num_of_dimensions(self):
        return 3

    