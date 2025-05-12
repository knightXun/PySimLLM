class BasicLogicalTopology:
    class BasicTopology:
        Ring = "Ring"

    def __init__(self, topology_type):
        self.topology_type = topology_type


class RingTopology(BasicLogicalTopology):
    class Direction:
        Clockwise = 1
        Anticlockwise = 2

    class Dimension:
        Local = 1
        Vertical = 2
        Horizontal = 3
        NA = 4

    def __init__(self, dimension, id, total_nodes_in_ring, index_in_ring, offset):
        super().__init__(BasicLogicalTopology.BasicTopology.Ring)
        self.name = "local"
        if dimension == self.Dimension.Vertical:
            self.name = "vertical"
        elif dimension == self.Dimension.Horizontal:
            self.name = "horizontal"
        if id == 0:
            print(f"ring of node 0, id: {id} dimension: {self.name} total nodes in ring: {total_nodes_in_ring} index in ring: {index_in_ring} offset: {offset} total nodes in ring: {total_nodes_in_ring}")
        self.id = id
        self.total_nodes_in_ring = total_nodes_in_ring
        self.index_in_ring = index_in_ring
        self.offset = offset
        self.dimension = dimension
        self.id_to_index = {id: index_in_ring}
        self.find_neighbors()

    def find_neighbors(self):
        self.next_node_id = self.id + self.offset
        if self.index_in_ring == self.total_nodes_in_ring - 1:
            self.next_node_id -= (self.total_nodes_in_ring * self.offset)
            assert self.next_node_id >= 0
        self.previous_node_id = self.id - self.offset
        if self.index_in_ring == 0:
            self.previous_node_id += (self.total_nodes_in_ring * self.offset)
            assert self.previous_node_id >= 0

    def get_receiver_node(self, node_id, direction):
        assert node_id in self.id_to_index
        index = self.id_to_index[node_id]
        if direction == self.Direction.Clockwise:
            receiver = node_id + self.offset
            if index == self.total_nodes_in_ring - 1:
                receiver -= (self.total_nodes_in_ring * self.offset)
                index = 0
            else:
                index += 1
            if receiver < 0:
                print(f"at dim: {self.name} at id: {self.id} dimension: {self.name} index: {index} ,node id: {node_id} ,offset: {self.offset} ,index_in_ring: {self.index_in_ring} receiver: {receiver}")
            assert receiver >= 0
            self.id_to_index[receiver] = index
            return receiver
        else:
            receiver = node_id - self.offset
            if index == 0:
                receiver += (self.total_nodes_in_ring * self.offset)
                index = self.total_nodes_in_ring - 1
            else:
                index -= 1
            if receiver < 0:
                print(f"at dim: {self.name} at id: {self.id} dimension: {self.name} index: {index} ,node id: {node_id} ,offset: {self.offset} ,index_in_ring: {self.index_in_ring} receiver: {receiver}")
            assert receiver >= 0
            self.id_to_index[receiver] = index
            return receiver

    def get_sender_node(self, node_id, direction):
        assert node_id in self.id_to_index
        index = self.id_to_index[node_id]
        if direction == self.Direction.Anticlockwise:
            sender = node_id + self.offset
            if index == self.total_nodes_in_ring - 1:
                sender -= (self.total_nodes_in_ring * self.offset)
                index = 0
            else:
                index += 1
            if sender < 0:
                print(f"at dim: {self.name} at id: {self.id} index: {index} ,node id: {node_id} ,offset: {self.offset} ,index_in_ring: {self.index_in_ring} ,sender: {sender}")
            assert sender >= 0
            self.id_to_index[sender] = index
            return sender
        else:
            sender = node_id - self.offset
            if index == 0:
                sender += (self.total_nodes_in_ring * self.offset)
                index = self.total_nodes_in_ring - 1
            else:
                index -= 1
            if sender < 0:
                print(f"at dim: {self.name} at id: {self.id} index: {index} ,node id: {node_id} ,offset: {self.offset} ,index_in_ring: {self.index_in_ring} ,sender: {sender}")
            assert sender >= 0
            self.id_to_index[sender] = index
            return sender

    def get_nodes_in_ring(self):
        return self.total_nodes_in_ring

    def is_enabled(self):
        tmp_index = self.index_in_ring
        tmp_id = self.id
        while tmp_index > 0:
            tmp_index -= 1
            tmp_id -= self.offset
        if tmp_id == 0:
            return True
        return False

    def get_num_of_nodes_in_dimension(self, dimension):
        return self.get_nodes_in_ring()
    