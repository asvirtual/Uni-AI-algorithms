from network import Network
from node import Node
from cpt import CPT


nodes = [
    Node(
        "Cloudy",
        cpt = {
            "Cloudy": [0.5, 0.5]
        }
    )
]

nodes.append(
    Node(
        "Sprinkler",
        parents = [nodes[0]],
        cpt = {
            "Cloudy": [
                [0.9, 0.1],
                [0.5, 0.5]
            ]
        }
    )
)

nodes.append(
    Node(
        "Rain",
        parents = [nodes[0]],
        cpt = {
            "Cloudy": [
                [0.2, 0.8],
                [0.9, 0.1]
            ]
        }
    )
)

nodes.append(
    Node(
        "Wet Grass",
        parents = [nodes[1], nodes[2]],
        cpt = {
            "Sprinkler": [
                { "Rain": [
                    [0.05, 0.95],
                    [0.10, 0.90]
                ] },
                { "Rain": [
                    [0.10, 0.90],
                    [0.90, 0.10]
                ] }
            ]
        }
    )
)

nodes[0].add_child(nodes[1])
nodes[0].add_child(nodes[2])
nodes[1].add_child(nodes[3])
nodes[2].add_child(nodes[3])

network = Network(nodes)
# for _ in range(10):
result = network.gibbs_sampling(nodes[2], { "Sprinkler": True, "Wet Grass": True }, iterations=10000)
print(result)