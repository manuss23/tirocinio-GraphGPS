import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from utils import typed_directed_edges

K = 10
VAL_RATIO = 0.2
percentages = [8, 16, 32, 64]
settings = ["percented_train_r", "percented_train_rs", "percented_train_rs_percented_s_50"]
TUTOR_DIR = "datasets/SocialNetwork/user_classification/tutor"
REAL_EDG = "datasets/SocialNetwork/raw/real_social_network.edg"
SYNTHETIC_EDG = "datasets/SocialNetwork/raw/synthetic_social_network.edg"

embeddings = torch.load(os.path.join(TUTOR_DIR, "users_bert_embs_rs.pt"), weights_only=False)
num_features = len(next(iter(embeddings.values())))
print(f"Embedding caricati: {len(embeddings)} utenti, {num_features} feature")

def read_pairs(edg_path):
    pairs = []
    with open(edg_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

def build_edges(pairs, nodes_mapping):
    sources = []
    targets = []
    for a, b in pairs:
        if a in nodes_mapping and b in nodes_mapping:
            sources.append(nodes_mapping[a])
            targets.append(nodes_mapping[b])
    return torch.tensor([sources, targets], dtype=torch.long)

real_pairs = read_pairs(REAL_EDG)
synthetic_pairs = read_pairs(SYNTHETIC_EDG)
real_network_users = sorted({u for pair in real_pairs for u in pair})
print(f"Rete reale: {len(real_pairs)} archi, {len(real_network_users)} utenti")
print(f"Rete sintetica: {len(synthetic_pairs)} archi")

for setting in settings:
    for percentage in percentages:
        for k in range(K):
            fold_dir = os.path.join(TUTOR_DIR, setting, f"{percentage}_perc", "cross_validation", f"fold_{k+1}")
            train_df = pd.read_csv(os.path.join(fold_dir, "train.tsv"), sep="\t", dtype={"account_id": str}, usecols=["account_id", "label"])
            test_df = pd.read_csv(os.path.join(fold_dir, "test.tsv"), sep="\t", dtype={"account_id": str}, usecols=["account_id", "label"])

            labeled_users = set(train_df["account_id"]) | set(test_df["account_id"])
            extra_users = [u for u in real_network_users if u not in labeled_users]
            extra_df = pd.DataFrame({"account_id": extra_users, "label": -1})
            users_df = pd.concat([train_df, test_df, extra_df], ignore_index=True)
            nodes_mapping = {account_id: i for i, account_id in enumerate(users_df["account_id"])}
            assert len(nodes_mapping) == len(users_df), "Utenti ripetuti"
            num_nodes = len(users_df)

            x = torch.zeros(num_nodes, num_features)
            missing = 0
            for account_id, i in nodes_mapping.items():
                if account_id in embeddings:
                    x[i] = embeddings[account_id].cpu()
                else:
                    missing += 1
            y = torch.tensor(users_df["label"].values, dtype=torch.long)

            edge_index = torch.cat([
                build_edges(real_pairs, nodes_mapping),
                build_edges(synthetic_pairs, nodes_mapping)
            ], dim=1)
            edge_index, edge_attr = typed_directed_edges(edge_index)

            real_df = train_df[train_df["account_id"].str.startswith("r")]
            synthetic_df = train_df[train_df["account_id"].str.startswith("s")]
            train_part, val_part = train_test_split(
                real_df,
                test_size=VAL_RATIO,
                stratify=real_df["label"],
                random_state=42
            )
            train_part = pd.concat([train_part, synthetic_df])

            train_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
            for split_df, mask in [(train_part, train_mask), (val_part, val_mask), (test_df, test_mask)]:
                idx = torch.tensor([nodes_mapping[a] for a in split_df["account_id"]], dtype=torch.long)
                mask[idx, k] = True

            tr = train_mask[:, k]
            va = val_mask[:, k]
            te = test_mask[:, k]
            in_split = tr.long() + va.long() + te.long()
            is_labeled = y != -1
            assert (in_split[is_labeled] == 1).all(), "Ogni nodo etichettato deve stare in un solo split"
            assert (in_split[~is_labeled] == 0).all(), "Nodi senza etichetta dentro una maschera"
            is_synthetic = torch.tensor([a.startswith("s") for a in users_df["account_id"]])
            assert not (va[is_synthetic] | te[is_synthetic]).any(), "Sintetici in validation o test"
            train_test_edges = (te[edge_index[0]] & tr[edge_index[1]]).sum().item()
            assert train_test_edges > 0, "Il test è ancora scollegato dal train"

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
            data.validate(raise_on_error=True)
            torch.save(data, os.path.join(fold_dir, "graph_directed.pt"))

            print(f"{setting} {percentage}% fold {k+1}:   Nodi: {num_nodes}   Archi: {edge_index.size(1)}   "
                  f"Train: {tr.sum().item()}   Val: {va.sum().item()}   Test: {te.sum().item()}   "
                  f"Archi test-train: {train_test_edges}   Senza embedding: {missing}")