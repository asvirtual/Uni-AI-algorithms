class Edge:
    def __init__(self, origin_node, destination_node, cost=1):
        self._origin_node = origin_node
        self._destination_node = destination_node
        self._cost = cost

    
    def get_origin_node(self):
        return self._origin_node
    

    def get_destination_node(self):
        return self._destination_node
    

    def get_cost(self):
        return self._cost
    

    def __str__(self):
        return f"{self._origin_node.get_label()} -> {self._destination_node.get_label()} ({self._cost})"