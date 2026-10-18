import torch
from sklearn.model_selection import StratifiedKFold

K = 10

# Caricamento del grafo orientato costruito da build_dataset.py
input_path = "datasets/SocialNetwork/processed_graph_directed.pt"
data = torch.load(input_path, weights_only=False)
print(f"Grafo caricato: {data}")

# Nodi etichettati
y = data.y
labeled_idx = (y != -1).nonzero(as_tuple=True)[0].numpy()
labeled_y = y[labeled_idx].numpy()
print(f"Nodi etichettati: {len(labeled_idx)}")

# Divisione in K fold stratificati
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
folds = []

for _, fold_pos in skf.split(labeled_idx, labeled_y):
    folds.append(labeled_idx[fold_pos])

for i in range(K):
    print(f"Fold {i}: {len(folds[i])} nodi")


# Creazione maschere con una colonna per ogni split
num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, K, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, K, dtype=torch.bool)

for i in range(K):
    test_fold = i
    val_fold = (i + 1) % K

    for j in range(K):
        if j == test_fold:
            test_mask[folds[j], i] = True
        elif j == val_fold:
            val_mask[folds[j], i] = True
        else:
            train_mask[folds[j], i] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# Controlli
for i in range(K):
    tr = train_mask[:, i]
    va = val_mask[:, i]
    te = test_mask[:, i]

    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any(), f"Split {i}: maschere sovrapposte!"
    assert (tr | va | te).sum() == len(labeled_idx), f"Split {i}: mancano nodi etichettati!"
    print(f"Split {i}:  Train: {tr.sum().item()}    Val: {va.sum().item()}  Test: {te.sum().item()}")

assert (test_mask[labeled_idx].sum(dim=1) == 1).all(), "Ogni nodo etichettato deve finire nel test una sola volta!"
data.validate(raise_on_error=True)
print("Controlli superati.")

# Salvataggio
output_path = "datasets/SocialNetwork/processed_graph_directed_cv.pt"
torch.save(data, output_path)
print(f"Grafo con split CV salvato in {output_path}")