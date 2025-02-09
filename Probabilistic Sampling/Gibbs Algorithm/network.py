import numpy as np
from node import Node


class Network:
    def __init__(self, nodes=None):
        if nodes is not None and not isinstance(nodes, list) and not all([isinstance(node, Node) for node in nodes]):
            raise ValueError("[ERROR] Nodes must be a list of nodes")

        self._nodes = nodes if nodes is not None else []
        self._nodes_map = { node.get_label(): node for node in self._nodes }

    
    def add_node(self, node):
        if not isinstance(node, Node):
            raise ValueError("[ERROR] Node must be an instance of Node")

        if node.get_label() in self._nodes_map:
            raise ValueError(f"[ERROR] Node {node.get_label()} already in network")

        self._nodes.append(node)
        self._nodes_map[node.get_label()] = node


    def get_node(self, label):
        return self._nodes_map[label]
    

    def gibbs_sampling(self, query_node, evidence, iterations=100):
        if not isinstance(query_node, Node):
            raise ValueError("[ERROR] Query node must be an instance of Node")
        
        if not isinstance(evidence, dict):
            raise ValueError("[ERROR] Evidence must be a dictionary")

        counts = np.zeros(len(query_node.get_values()))
        non_evidence_nodes = [node for node in self._nodes if node.get_label() not in evidence]
        states = []
        state = { node.get_label(): node.random_sample() for node in non_evidence_nodes }
        for label, value in evidence.items(): state[label] = value

        def filter_evidence(node, state):
            node_evidence = {}
            parents_labels = [parent.get_label() for parent in node.get_parents()]
            
            for label in state: 
                if label in parents_labels:
                    node_evidence[label] = state[label]
            
            return node_evidence

        for _ in range(iterations):
            # Sample each node that is not evidence
            node = np.random.choice(non_evidence_nodes)
            # Filter evidence for the node (only values for its parents)
            node_evidence = filter_evidence(node, state)

            # Prepare the probabilities for each value of the node
            probabilities = []
            for idx, value in enumerate(node.get_values()):
                # cpt = node_evidence.copy()
                # P(X = x | MB(X)) ∝ P(X = x | Parents(X)) × ∏₍child ∈ Children(X)₎ P(child value | Parents(child) with X set to x)
                # Extract P(X = x | Parents(X)) from the node's cpt
                prob = node.get_cpt()[node_evidence][idx]

                # Calculate the productory of the children nodes given evidence for their parents 
                # ∏₍child ∈ Children(X)₎ P(child value | Parents(child) with X set to x)
                for child in node.get_next_child():
                    # Filter evidence for the child node (only evidence for its parents)
                    child_node_evidence = filter_evidence(child, state)
                    # We need to include the value of the node (the one we are currently considering) in the evidence for the child
                    # in order to sample a value for the child node
                    child_node_evidence.update({ node.get_label(): value })
                    # Extract P(child value | Parents(child) with X set to x) from the cpt of the child node
                    prob *= child.get_cpt()[child_node_evidence][child.get_values().index(state[child.get_label()])]
                
                probabilities.append(prob)
            
            probabilities = np.array(probabilities)
            probabilities /= np.sum(probabilities)
            # Sample a value for the node                
            sampled_value = np.random.choice(node.get_values(), p=probabilities)
            # Update the state with the sampled value
            state[node.get_label()] = sampled_value

            # At each iteration, save the state of the network
            states.append(state)
            state = state.copy()

            '''
                P(Cloudy | Sprinkler = s, Rain = ¬r) will be proportional to P(Sprinkler=s | Cloudy) * P(Rain = ¬r | Cloudy) * P(Cloudy)
            '''

        # Calculate the probabilities for the query node
        result = np.zeros(len(query_node.get_values()))
        # For each possible value of the query node
        for i, value in enumerate(query_node.get_values()):
            # Calculate the probability of the specific value of the query node given the evidence by counting the number of times
            # the value and the query evidence appear in the states 

            result[i] = sum([1 for state in states if state[query_node.get_label()] == value])
            # And then divide by the total number of states
            result[i] /= len(states)


        return result


    def __get_item__(self, label):
        return self.get_node(label)
