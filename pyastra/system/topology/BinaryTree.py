import enum 

from Node import Node
from BasicLogicalTopology import BasicLogicalTopology


class BinaryTree(BasicLogicalTopology):
    class TreeType(enum.Enum):
        RootMax = 0
        RootMin = 1

    class Type(enum.Enum):
        Leaf = 0
        Root = 1
        Intermediate = 2

    def __init__(self, id, tree_type, total_tree_nodes, start, stride):
        self.total_tree_nodes = total_tree_nodes
        self.start = start
        self.tree_type = tree_type
        self.stride = stride
        self.tree = Node(-1)
        depth = 1
        tmp = total_tree_nodes
        while tmp > 1:
            depth += 1
            tmp //= 2
        if self.tree_type == BinaryTree.TreeType.RootMin:
            self.tree.right_child = self.initialize_tree(depth - 1, self.tree)
        else:
            self.tree.left_child = self.initialize_tree(depth - 1, self.tree)
        self.node_list = {}
        self.build_tree(self.tree)

    def __del__(self):
        pass

    def initialize_tree(self, depth, parent):
        tmp = Node(-1, parent)
        if depth > 1:
            tmp.left_child = self.initialize_tree(depth - 1, tmp)
            tmp.right_child = self.initialize_tree(depth - 1, tmp)
        return tmp

    def build_tree(self, node):
        if node.left_child is not None:
            self.build_tree(node.left_child)
        node.id = self.start
        self.node_list[self.start] = node
        self.start += self.stride
        if node.right_child is not None:
            self.build_tree(node.right_child)

    def get_parent_id(self, id):
        parent = self.node_list[id].parent
        if parent is not None:
            return parent.id
        return -1

    def get_right_child_id(self, id):
        child = self.node_list[id].right_child
        if child is not None:
            return child.id
        return -1

    def get_left_child_id(self, id):
        child = self.node_list[id].left_child
        if child is not None:
            return child.id
        return -1

    def get_node_type(self, id):
        node = self.node_list[id]
        if node.parent is None:
            return BinaryTree.Type.Root
        elif node.left_child is None and node.right_child is None:
            return BinaryTree.Type.Leaf
        else:
            return BinaryTree.Type.Intermediate

    def is_enabled(self, id):
        return id % self.stride == 0

    def print(self, node):
        print(f"I am node: {node.id}", end="")
        if node.left_child is not None:
            print(f" and my left child is: {node.left_child.id}", end="")
        if node.right_child is not None:
            print(f" and my right child is: {node.right_child.id}", end="")
        if node.parent is not None:
            print(f" and my parent is: {node.parent.id}", end="")
        typ = self.get_node_type(node.id)
        if typ == BinaryTree.Type.Root:
            print(" and I am Root ", end="")
        elif typ == BinaryTree.Type.Intermediate:
            print(" and I am Intermediate ", end="")
        elif typ == BinaryTree.Type.Leaf:
            print(" and I am Leaf ", end="")
        print()
        if node.left_child is not None:
            self.print(node.left_child)
        if node.right_child is not None:
            self.print(node.right_child)    