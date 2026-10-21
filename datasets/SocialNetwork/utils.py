def read_edges(edg_path="datasets/SocialNetwork/raw/real_social_network.edg"):
    nodes_mapping = {}
    current_index = 0
    sources = []
    targets = []

    with open(edg_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            source_node, target_node = parts

            if source_node not in nodes_mapping:
                nodes_mapping[source_node] = current_index
                current_index += 1

            if target_node not in nodes_mapping:
                nodes_mapping[target_node] = current_index
                current_index += 1

            sources.append(nodes_mapping[source_node])
            targets.append(nodes_mapping[target_node])

    return nodes_mapping, sources, targets


def load_nodes_mapping(edg_path="datasets/SocialNetwork/raw/real_social_network.edg"):
    nodes_mapping, _, _ = read_edges(edg_path)
    return nodes_mapping