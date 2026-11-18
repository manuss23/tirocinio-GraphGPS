import pandas as pd
import torch
from torch_geometric.utils import remove_self_loops, coalesce

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


def load_rs_mapping(labels_path="datasets/SocialNetwork/raw/synthetic_users_labels.tsv"):
    nodes_mapping = load_nodes_mapping()
    synthetic_df = pd.read_csv(labels_path, sep="\t", dtype=str)

    for account in synthetic_df["account_id"]:
        account = account.strip()
        if account not in nodes_mapping:
            nodes_mapping[account] = len(nodes_mapping)

    return nodes_mapping


def typed_directed_edges(edge_index):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index)
    num_relations = edge_index.size(1)

    follower = edge_index[0]
    followed = edge_index[1]

    edge_index_directed = torch.cat([
        torch.stack([followed, follower]),
        torch.stack([follower, followed])
    ], dim=1)
    edge_attr = torch.cat([
        torch.zeros(num_relations, dtype=torch.long),
        torch.ones(num_relations, dtype=torch.long)
    ])

    return edge_index_directed, edge_attr