class Node:
    def __init__(self, label, heuristic=1):
        self._label = label
        self._heuristic = heuristic
        self._cost = 0


    def get_label(self):
        return self._label
    

    def get_heuristic(self):
        return self._heuristic
    

    def get_cost(self):
        return self._cost
    

    def set_cost(self, cost):
        self._cost = cost


    def __str__(self):
        return f"{self._label}, cost: {self._cost}, heuristic: {self._heuristic}"
    

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise ValueError(f"[ERROR] Can't compare Node {self} and {type(other)} {other}")

        return self._heuristic == other.get_heuristic() and self._label == other.get_label()
    