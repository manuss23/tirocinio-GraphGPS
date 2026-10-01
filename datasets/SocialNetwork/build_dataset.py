import torch
from torch_geometric.utils import remove_self_loops, to_undirected
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Inizializzazione strutture dati
nodes_mapping = {}
current_index = 0
sources = []
targets = []

edg_directory = "datasets/SocialNetwork/raw/real_social_network.edg"

# Lettura topologia e mappatura nodi
with open(edg_directory, "r") as f:
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

print(f"Mappatura completata. Trovati {len(nodes_mapping)} nodi univoci.")


# Creazione tensore edge_index
edge_index = torch.tensor([sources, targets], dtype=torch.long)
print(f"Tensore edge_index generato con forma: {edge_index.shape}")

# Pulizia archi
edge_index, _ = remove_self_loops(edge_index)
edge_index = to_undirected(edge_index)
print(f"edge_index dopo la pulizia: {edge_index.shape}")


# Estrazione dei testi e raggruppamento testi per nodo
tsv_directory = "datasets/SocialNetwork/raw/real_posts.tsv"
df_posts = pd.read_csv(tsv_directory, sep='\t', dtype=str)

posts_per_node = {i: [] for i in range(len(nodes_mapping))}

for _, row in df_posts.iterrows():
    account_id = row['account_id'].strip()
    content = row['content']

    if pd.isna(content) or not content.strip() or account_id not in nodes_mapping:
        continue

    posts_per_node[nodes_mapping[account_id]].append(content)


# Vettorizzazione dei testi e tensore x
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
x_list = []

for i in range(len(nodes_mapping)):
    texts = posts_per_node[i]

    if len(texts) > 0:
        embeddings = model.encode(texts, convert_to_tensor=True)
        x_list.append(embeddings.mean(dim=0).cpu())
    else:
        x_list.append(torch.zeros(384))

x = torch.stack(x_list)
print(f"Tensore x generato con forma: {x.shape}")


# Estrazione etichette e tensore y
labels_directory = "datasets/SocialNetwork/raw/real_users_labels.tsv"
df_labels = pd.read_csv(labels_directory, sep='\t', dtype=str)

y_list = [-1] * len(nodes_mapping)

for _, row in df_labels.iterrows():
    account_id = row['account_id'].strip()

    if account_id in nodes_mapping:
        y_list[nodes_mapping[account_id]] = int(row['label'])

y = torch.tensor(y_list, dtype=torch.long)
print(f"Tensore y generato con forma: {y.shape}")
print(f"Nodi etichettati: {(y != -1).sum().item()} su {len(nodes_mapping)}")


# Split stratificato 70 / 15 / 15 sui nodi etichettati
labeled_idx = (y != -1).nonzero(as_tuple=True)[0].numpy()
labeled_y = y[labeled_idx].numpy()

idx_train, idx_temp, y_train, y_temp = train_test_split(
    labeled_idx, labeled_y,
    test_size=0.30,
    random_state=42,
    stratify=labeled_y
)

idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Creazione maschere
num_nodes = len(nodes_mapping)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

print(f"Train: {train_mask.sum().item()}    Val: {val_mask.sum().item()}    Test: {test_mask.sum().item()}")


# Assemblaggio oggetto Data
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

print("\nOggetto Data finale:")
print(data)


# Controlli
data.validate(raise_on_error=True)
print(f"Grafo non orientato: {data.is_undirected()}")
print(f"Self-loop presenti: {data.has_self_loops()}")
print(f"Nodi isolati presenti: {data.has_isolated_nodes()}")

assert not (train_mask & val_mask).any() and not (train_mask & test_mask).any() and not (val_mask & test_mask).any(), "Le mask si sovrappongono!"
assert (y[train_mask | val_mask | test_mask] != -1).all(), "Nodi senza label nelle mask!"
print("Controlli superati.")


# Salvataggio grafo
output_path = "datasets/SocialNetwork/processed_graph.pt"
torch.save(data, output_path)
print(f"Grafo salvato in {output_path}")