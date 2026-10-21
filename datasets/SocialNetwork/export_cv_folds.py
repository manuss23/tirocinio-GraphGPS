import os
import torch
import pandas as pd
from utils import load_nodes_mapping

K = 10
BASE_DIR = "datasets/SocialNetwork/user_classification/cross_validation"


data = torch.load("datasets/SocialNetwork/processed_graph_directed_cv.pt", weights_only=False)

modes_mapping = load_nodes_mapping()

index_to_account = {index: account for account, index in nodes_mapping.items()}
assert len(index_to_account) == data.num_nodes, "La mappatura non corrisponde al grafo"

df_posts = pd.read_csv("datasets/SocialNetwork/raw/real_posts.tsv", sep='\t', dtype=str)
df_posts = df_posts[["account_id", "id", "content", "exact_level_found"]]
df_posts["account_id"] = df_posts["account_id"].str.strip()

for i in range(K):
    fold_dir = os.path.join(BASE_DIR, f"fold_{i+1}")
    os.makedirs(fold_dir, exist_ok=True)

    for split_name, mask in [("train", data.train_mask[:, i] | data.val_mask[:, i]), ("test", data.test_mask[:, i])]:
        node_idx = mask.nonzero(as_tuple=True)[0].tolist()

        users_df = pd.DataFrame({
            "account_id": [index_to_account[n] for n in node_idx],
            "label": [data.y[n].item() for n in node_idx]
        })
        posts_df = df_posts[df_posts["account_id"].isin(set(users_df["account_id"]))]

        users_df.to_csv(os.path.join(fold_dir, f"{split_name}.tsv"), sep="\t", index=False)
        posts_df.to_csv(os.path.join(fold_dir, f"{split_name}_posts.tsv"), sep="\t", index=False)

        print(f"Fold {i + 1} - {split_name}: {len(users_df)} utenti, {len(posts_df)} post")