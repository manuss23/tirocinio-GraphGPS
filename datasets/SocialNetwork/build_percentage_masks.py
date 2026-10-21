import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from utils import load_nodes_mapping

K = 10
VAL_RATIO = 0.2
percentages = [80, 60, 40, 20]
setting = "r"
SPLIT_DIR = "datasets/SocialNetwork/user_classification"

data = torch.load("datasets/SocialNetwork/processed_graph_directed.pt", weights_only=False)
num_nodes = data.num_nodes

nodes_mapping = load_nodes_mapping()
assert len(nodes_mapping) == num_nodes, "La mappatura non corrisponde al grafo"

for percentage in percentages:
    train_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, K, dtype=torch.bool)

    for i in range(K):
        fold_dir = os.path.join(SPLIT_DIR, f"percented_train_{setting}", f"{percentage}_perc", "cross_validation", f"fold_{i+1}")
        train_df = pd.read_csv(os.path.join(fold_dir, "train.tsv"), sep="\t", dtype={"account_id": str})
        test_df = pd.read_csv(os.path.join(fold_dir, "test.tsv"), sep="\t", dtype={"account_id": str})

        train_part, val_part = train_test_split(
            train_df,
            test_size=VAL_RATIO,
            stratify=train_df["label"],
            random_state=42
        )

        for split_df, mask in [(train_part, train_mask), (val_part, val_mask), (test_df, test_mask)]:
            idx = torch.tensor([nodes_mapping[a] for a in split_df["account_id"]], dtype=torch.long)
            assert (data.y[idx] == torch.tensor(split_df["label"].values)).all(), "Etichette non coerenti"
            mask[idx, i] = True

        tr = train_mask[:, i]
        va = val_mask[:, i]
        te = test_mask[:, i]
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any(), print(f"{percentage}% - Fold {i + 1}:   Train: {tr.sum().item()}    Val: {va.sum().item()}  Test: {te.sum().item()}")

    labeled = data.y != -1
    assert (test_mask[labeled].sum(dim=1) == 1).all(), "Ogni nodo etichettato deve finire nel test una sola volta"

    out_data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    out_data.validate(raise_on_error=True)

    output_path = os.path.join(SPLIT_DIR, f"percented_train_{setting}", f"{percentage}_perc", "graph_directed.pt")
    torch.save(out_data, output_path)
    print(f"Salvato {output_path}\n")