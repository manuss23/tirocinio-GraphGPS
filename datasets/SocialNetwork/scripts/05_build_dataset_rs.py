import torch
import pandas as pd
from torch_geometric.utils import remove_self_loops, coalesce
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from utils import load_rs_mapping

data = torch.load("datasets/SocialNetwork/graphs/processed_graph_directed.pt", weights_only=False)
num_real = data.num_nodes

nodes_mapping = load_rs_mapping()
num_nodes = len(nodes_mapping)
num_synthetic = num_nodes - num_real
print(f"Nodi reali: {num_real}    Nodi sintetici: {num_synthetic}")

sources = []
targets = []
with open("datasets/SocialNetwork/raw/synthetic_social_network.edg", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        sources.append(nodes_mapping[parts[0]])
        targets.append(nodes_mapping[parts[1]])

edge_index = torch.tensor([sources, targets], dtype=torch.long)
edge_index, _ = remove_self_loops(edge_index)
edge_index = coalesce(edge_index)
num_relations = edge_index.size(1)

follower = edge_index[0]
followed = edge_index[1]
edge_index_synthetic = torch.cat([torch.stack([followed, follower]), torch.stack([follower, followed])], dim=1)
edge_attr_synthetic = torch.cat([torch.zeros(num_relations, dtype=torch.long), torch.ones(num_relations, dtype=torch.long)])
print(f"Relazioni sintetiche: {num_relations}")

df_posts = pd.read_csv("datasets/SocialNetwork/raw/synthetic_posts.tsv", sep="\t", dtype=str)
posts_per_node = {i: [] for i in range(num_real, num_nodes)}

for _, row in df_posts.iterrows():
    account_id = row["account_id"].strip()
    content = row["content"]
    if pd.isna(content) or not content.strip() or account_id not in nodes_mapping:
        continue
    posts_per_node[nodes_mapping[account_id]].append(content)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
x_list = []
for i in range(num_real, num_nodes):
    texts = posts_per_node[i]
    if len(texts) > 0:
        x_list.append(model.encode(texts, convert_to_tensor=True).mean(dim=0).cpu())
    else:
        x_list.append(torch.zeros(384))
x_synthetic = torch.stack(x_list)


df_labels = pd.read_csv("datasets/SocialNetwork/raw/synthetic_users_labels.tsv", sep="\t", dtype=str)
y_synthetic = torch.full((num_synthetic,), -1, dtype=torch.long)
for _, row in df_labels.iterrows():
    y_synthetic[nodes_mapping[row["account_id"].strip()] - num_real] = int(row["label"])


data_rs = Data(
    x=torch.cat([data.x, x_synthetic]),
    edge_index=torch.cat([data.edge_index, edge_index_synthetic], dim=1),
    edge_attr=torch.cat([data.edge_attr, edge_attr_synthetic]),
    y=torch.cat([data.y, y_synthetic])
)
data_rs.validate(raise_on_error=True)

assert (data_rs.x[:num_real] == data.x).all() and (data_rs.y[:num_real] == data.y).all(), "Parte reale modificata"
assert (data_rs.y[num_real:] != -1).all(), "Sintetici senza etichetta"
print(data_rs)

torch.save(data_rs, "datasets/SocialNetwork/graphs/processed_graph_directed_rs.pt")
print("Salvato processed_graph_directed_rs.pt")