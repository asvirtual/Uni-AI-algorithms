from node import Node
from edge import Edge
import math


def astar_search(start, goal, edges):
    if not isinstance(start, Node) or not isinstance(goal, Node):
        raise ValueError(f"[ERROR] Start and goal must be of type Node, got {type(start)} and {type(goal)}")
    
    if goal.get_heuristic() != 0:
        raise ValueError(f"[ERROR] Inadmissible heuristic: heuristic(goal) must be 0, got {goal.get_heuristic()}")
    
    # path = []
    contour = [start]
    current = start
    while current != goal:
        # Find all edges that have the current node as origin
        node_edges = [edge for edge in edges if edge.get_origin_node() == current]
        for edge in node_edges:
            # Compute the cost of the destination nodes as the current node's cost + the edge cost
            # Update the cost of the destination node and add it to the contour if it's lower than the current cost or if it hasn't been visited yet
            destination_node = edge.get_destination_node()
            if destination_node == start: continue

            if destination_node.get_cost() == 0 or destination_node.get_cost() > current.get_cost() + edge.get_cost():
                destination_node.set_cost(current.get_cost() + edge.get_cost())
                contour.append(destination_node)
            
        # Remove the current node from the contour as it has been expanded and add it to the expanded nodes
        contour.remove(current)

        # Select the node with the lowest cost + heuristic from the contour
        best_contour_cost = math.inf
        for node in contour:
            node_cost = node.get_cost() + node.get_heuristic()
            if node_cost < best_contour_cost:
                best_contour_cost = node_cost
                current = node

        # path.append(current)

    # Reconstruct the path from the start to the goal
    path = [goal]
    while current != start:
        # Find the edge that connects the current node to its predecessor
        node_edges = [edge for edge in edges if edge.get_destination_node() == current]
        # Find the predecessor node by checking which node has a cost equal to the current node's cost - the edge cost
        # This is the node that was expanded to reach the current node and it's the optimal predecessor
        predecessor = list(
            filter(
                lambda e: e.get_origin_node().get_cost() + e.get_cost() == current.get_cost(), 
                node_edges
            )
        )[0].get_origin_node()

        # Add the predecessor to the head of the list
        path.insert(0, predecessor)
        # Move to the predecessor
        current = predecessor
    
    return path


nodes = [
    Node("Arad", 366),
    Node("Zerind", 374),
    Node("Oradea", 380),
    Node("Sibiu", 253),
    Node("Timisoara", 329),
    Node("Lugoj", 244),
    Node("Mehadia", 241),
    Node("Drobeta", 242),
    Node("Craiova", 160),
    Node("Rimnicu Vilcea", 193),
    Node("Fagaras", 176),
    Node("Pitesti", 98),
    Node("Bucharest", 0),
    Node("Giurgiu", 77),
    Node("Urziceni", 80),
    Node("Hirsova", 151),
    Node("Eforie", 161),
    Node("Vaslui", 199),
    Node("Iasi", 226),
    Node("Neamt", 234)
]

edges = [
    Edge(nodes[0], nodes[1], 75),
    Edge(nodes[0], nodes[3], 140),
    Edge(nodes[0], nodes[4], 118),
    Edge(nodes[1], nodes[2], 71),
    Edge(nodes[2], nodes[3], 151),
    Edge(nodes[3], nodes[9], 80),
    Edge(nodes[3], nodes[10], 99),
    Edge(nodes[4], nodes[5], 111),
    Edge(nodes[5], nodes[6], 70),
    Edge(nodes[6], nodes[7], 75),
    Edge(nodes[7], nodes[8], 120),
    Edge(nodes[8], nodes[9], 146),
    Edge(nodes[9], nodes[11], 97),
    Edge(nodes[10], nodes[12], 211),
    Edge(nodes[11], nodes[12], 101),
    Edge(nodes[12], nodes[13], 90),
    Edge(nodes[12], nodes[14], 85),
    Edge(nodes[14], nodes[15], 98),
    Edge(nodes[14], nodes[17], 142),
    Edge(nodes[15], nodes[16], 86),
    Edge(nodes[17], nodes[18], 92),
    Edge(nodes[18], nodes[19], 87)   
]

# Bi-directional graph
for edge in edges.copy():
    edges.append(Edge(edge.get_destination_node(), edge.get_origin_node(), edge.get_cost()))

# print(len(edges))
# for edge in edges:
#     print(edge)

best_path = astar_search(Node("Arad", 366), Node("Bucharest", 0), edges)
for edge in best_path:
    print(edge.get_label())

#  ------------------------------------------------
#  Previous implementation of the path reconstruction:

    # for edge in node_edges:
    #     origin_node = edge.get_origin_node()
    #     if origin_node.get_cost() + edge.get_cost() == current.get_cost():
    #         path.append(origin_node)
    #         current = origin_node
    #         break
                
# for node in path.copy():
#     node_edges = [edge for edge in edges if edge.get_origin_node() == node]
#     if len(node_edges) <= 1: continue
#     node_to_keep_cost = min(map(lambda n: n.get_cost(), [edge.get_destination_node() for edge in node_edges]))
#     for to_remove in [edge.get_destination_node() for edge in node_edges if edge.get_destination_node().get_cost() > node_to_keep_cost]:
#         path.remove(to_remove)