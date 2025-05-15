from enum import Enum

from ComplexLogicalTopology import ComplexLogicalTopology
from Common import CollectiveImplementationType

from RingTopology import RingTopology
from DoubleBinaryTreeTopology import DoubleBinaryTreeTopology

class GeneralComplexTopology(ComplexLogicalTopology):
    def __init__(self, id, dimension_size, collective_implementation):
        self.dimension_topology = []
        offset = 1
        last_dim = len(collective_implementation) - 1
        for dim in range(len(collective_implementation)):
            if collective_implementation[dim].type in [
                CollectiveImplementationType.Ring,
                CollectiveImplementationType.Direct,
                CollectiveImplementationType.HalvingDoubling,
                CollectiveImplementationType.NcclFlowModel,
                CollectiveImplementationType.NcclTreeFlowModel
            ]:
                ring = RingTopology(
                    RingTopology.Dimension.NA,
                    id,
                    dimension_size[dim],
                    (id % (offset * dimension_size[dim])) // offset,
                    offset
                )
                self.dimension_topology.append(ring)
            elif collective_implementation[dim].type in [
                CollectiveImplementationType.OneRing,
                CollectiveImplementationType.OneDirect,
                CollectiveImplementationType.OneHalvingDoubling
            ]:
                total_npus = 1
                for d in dimension_size:
                    total_npus *= d
                ring = RingTopology(
                    RingTopology.Dimension.NA, id, total_npus, id % total_npus, 1
                )
                self.dimension_topology.append(ring)
                return
            elif collective_implementation[dim].type == CollectiveImplementationType.DoubleBinaryTree:
                if dim == last_dim:
                    DBT = DoubleBinaryTreeTopology(
                        id, dimension_size[dim], id % offset, offset
                    )
                else:
                    DBT = DoubleBinaryTreeTopology(
                        id,
                        dimension_size[dim],
                        (id - (id % (offset * dimension_size[dim]))) + (id % offset),
                        offset
                    )
                self.dimension_topology.append(DBT)
            offset *= dimension_size[dim]

    def __del__(self):
        for topology in self.dimension_topology:
            del topology

    def get_basic_topology_at_dimension(self, dimension, type):
        return self.dimension_topology[dimension].get_basic_topology_at_dimension(0, type)

    def get_num_of_nodes_in_dimension(self, dimension):
        if dimension >= len(self.dimension_topology):
            print(f"dim: {dimension} requested! but max dim is: {len(self.dimension_topology) - 1}")
        assert dimension < len(self.dimension_topology)
        return self.dimension_topology[dimension].get_num_of_nodes_in_dimension(0)

    def get_num_of_dimensions(self):
        return len(self.dimension_topology)

    