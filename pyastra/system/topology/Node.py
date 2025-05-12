class Node:
    def __init__(self, id, parent=None, left_child=None, right_child=None):
        self.id = id
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child