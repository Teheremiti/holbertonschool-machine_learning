#!/usr/bin/env python3
""" Node, Leaf, and Decision_Tree classes """
import numpy as np


class Node:
    """ Defines a node of the decision tree. """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Class constructor for Node instances.

        Args:
            feature (int, optional): _description_. Defaults to None.
            threshold (float, optional): _description_. Defaults to None.
            left_child (Node, optional): _description_. Defaults to None.
            right_child (Node, optional): _description_. Defaults to None.
            is_root (bool, optional): _description_. Defaults to False.
            depth (int, optional): _description_. Defaults to 0.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Computes the depth of a decision tree using recursion. """
        def max_depth_recursion(node, depth):
            if node.is_leaf is True:
                return depth
            else:
                left_depth = max_depth_recursion(node.left_child, depth + 1)
                right_depth = max_depth_recursion(node.right_child, depth + 1)
                return max(left_depth, right_depth)

        return max_depth_recursion(self, 0)


class Leaf(Node):
    """ Defines a leaf of the decision tree. A leaf has no childs. """
    def __init__(self, value, depth=None):
        """
        Class constructor for Leaf instances.

        Args:
            value (int): The value held by the leaf.
            depth (int, optional): The depth of the leaf. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth


class Decision_Tree():
    """ Defines a decision tree. """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Class constructor for Decision_tree instances.

        Args:
            max_depth (int, optional): _description_. Defaults to 10.
            min_pop (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 0.
            split_criterion (str, optional): description. Defaults to "random".
            root (bool, optional): _description_. Defaults to None.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()
