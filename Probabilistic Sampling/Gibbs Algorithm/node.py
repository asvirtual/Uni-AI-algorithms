import numpy as np
from cpt import CPT


class Node:
    def __init__(self, label, parents=None, children=None, cpt=None, values=None):
        if parents is not None and not isinstance(parents, list) and not all([isinstance(parent, Node) for parent in parents]):
            raise ValueError("[ERROR] Parents must be a list of nodes")
        
        if children is not None and not isinstance(children, list) and not all([isinstance(child, Node) for child in children]):
            raise ValueError("[ERROR] Children must be a list of nodes")
        
        if cpt is not None and not isinstance(cpt, dict):
            raise ValueError("[ERROR] CPT must be a dictionary")

        self._label = label
        self._parents = parents if parents is not None else []
        self._children = children if children is not None else []
        self._cpt = CPT(cpt) if cpt is not None else CPT()
        self._values = values if values is not None else [True, False]

        for parent in self._parents:
            parent.add_child(self)

    
    def get_label(self):
        return self._label
    

    def get_values(self):
        return self._values
    

    def get_cpt(self):
        return self._cpt
    

    def add_child(self, child):
        if not isinstance(child, Node):
            raise ValueError("[ERROR] Child must be an instance of Node")
        
        if child in self._children:
            return
        
        self._children.append(child)
    

    def is_parent(self, node):
        return node in self._parents
    

    def is_child(self, node):
        return node in self._children
    

    def get_parents(self):
        return self._parents.copy()
    

    def get_next_child(self):
        yield from self._children


    def get_next_parent(self):
        yield from self._parents


    def get_markov_blanket(self):
        blanket = set(self._parents)
        for child in self._children:
            blanket.add(child)
            blanket.update(child.parents)
        
        return blanket
    

    def sample(self, evidence):
        probs = self._cpt[evidence]
        return np.random.choice(self._values, p=probs)
    

    def random_sample(self):
        return np.random.choice(self._values)
    

    def __str__(self):
        return f"Node {self._label}"

        