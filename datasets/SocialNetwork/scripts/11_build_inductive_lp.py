import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import degree
from utils import typed_directed_edges

K = 10
VAL_RATIO = 0.2
percentages = [8, 16, 32, 64]
settings = ["percented_train_r", "percented_train_rs", "percented_train_rs_percented_s_50"]
LP_DIR = "datasets/SocialNetwork/user_classification/percentages_8_64"

embeddings = torch.load(os.path.join(LP_DIR, "users_bert_embs_rs.pt"), weights_only=False)
num_features = len(next(iter(embeddings.values())))


def read_pairs(edg_path):
    pairs = []
    with open(edg_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

def build_graph(users_df, edg_path, train_ids, val_ids, test_ids):
    nodes_mapping = {a: i for i, a in enumerate(users_df["account_id"])}
    num_nodes = len(nodes_mapping)

    x = torch.zeros(num_nodes, num_features)
    missing = 0
    for account_id, i in nodes_mapping.items():
        if account_id in embeddings:
            x[i] = embeddings[account_id].cpu()
        else:
            missing += 1
    y = torch.tensor(users_df["label"].values, dtype=torch.long)

    sources, targets, dropped = [], [], 0
    for a, b in read_pairs(edg_path):
        if a in nodes_mapping and b in nodes_mapping:
            sources.append(nodes_mapping[a])
            targets.append(nodes_mapping[b])
        else:
            dropped += 1
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index, edge_attr = typed_directed_edges(edge_index)

    masks = []
    for ids in (train_ids, val_ids, test_ids):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        idx = [nodes_mapping[a] for a in ids if a in nodes_mapping]
        mask[idx] = True
        masks.append(mask)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=masks[0], val_mask=masks[1], test_mask=masks[2])
    return data, missing, dropped

for setting in settings:
    for percentage in percentages:
        for k in range(K):
            fold_dir = os.path.join(LP_DIR, setting, f"{percentage}_perc", "cross_validation", f"fold_{k+1}")
            train_df = pd.read_csv(os.path.join(fold_dir, "train.tsv"), sep="\t", dtype={"account_id": str}, usecols=["account_id", "label"])
            test_df = pd.read_csv(os.path.join(fold_dir, "test.tsv"), sep="\t", dtype={"account_id": str}, usecols=["account_id", "label"])

            real_df = train_df[train_df["account_id"].str.startswith("r")]
            synthetic_df = train_df[train_df["account_id"].str.startswith("s")]
            train_part, val_part = train_test_split(real_df, test_size=VAL_RATIO, stratify=real_df["label"], random_state=42)

            train_ids = set(train_part["account_id"]) | set(synthetic_df["account_id"])
            val_ids = set(val_part["account_id"])
            test_ids = set(test_df["account_id"])

            train_graph, miss_tr, drop_tr = build_graph(train_df, os.path.join(fold_dir, "train_network.edg"), train_ids, val_ids, test_ids)
            test_graph, miss_te, dorop_te = build_graph(test_df, os.path.join(fold_dir, "test_network.edg"), train_ids, val_ids, test_ids)

            assert not (train_ids & test_ids), "Utenti di test nel train"
            assert not train_graph.test_mask.any(), "Nodi di test nella rete di train"
            assert train_graph.train_mask.sum() == len(train_ids) and train_graph.val_mask.sum() == len(val_ids), "Train o val incompleti"
            assert test_graph.test_mask.all(), "Nella rete di test devono esserci solo nodi di test"
            assert not (test_graph.train_mask.any() or test_graph.val_mask.any()), "Train o val nella rete di test"
            assert (train_graph.y != -1).all() and (test_graph.y != -1).all(), "Nodi senza etichetta"
            train_graph.validate(raise_on_error=True)
            test_graph.validate(raise_on_error=True)

            torch.save([train_graph, test_graph], os.path.join(fold_dir, "graphs.pt"))

            test_deg = degree(test_graph.edge_index[0], test_graph.num_nodes)
            print(f"{setting} {percentage}% fold {k+1}:   "
                  f"Rete train: {train_graph.num_nodes} nodi, {train_graph.num_edges} archi "
                  f"(train {len(train_ids)}, val {len(val_ids)}), esclusi {drop_tr}   "
                  f"Rete test: {test_graph.num_nodes} nodi, {test_graph.num_edges} archi, esclusi {drop_tr}, "
                  f"isolati {(test_deg == 0).float().mean().item():.1%}   "
                  f"Senza embedding: {miss_tr + miss_te}")