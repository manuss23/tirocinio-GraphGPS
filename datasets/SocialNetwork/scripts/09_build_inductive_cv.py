import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, degree

K = 10
INPUT_PATH = "datasets/SocialNetwork/graphs/processed_graph_directed_cv.pt"
OUT_DIR = "datasets/SocialNetwork/user_classification/cross_validation"

data = torch.load(INPUT_PATH, weights_only=False)
print(f"Grafo caricato: {data}")

def induced_graph(nodes, train_mask, val_mask, test_mask):
    edge_index, edge_attr = subgraph(nodes, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)
    return Data(x=data.x[nodes], y=data.y[nodes], edge_index=edge_index, edge_attr=edge_attr,
                train_mask=train_mask[nodes], val_mask=val_mask[nodes], test_mask=test_mask[nodes])


for k in range(K):
    tr = data.train_mask[:, k]
    va = data.val_mask[:, k]
    te = data.test_mask[:, k]

    train_nodes = (tr | va).nonzero(as_tuple=True)[0]
    test_nodes = te.nonzero(as_tuple=True)[0]

    train_graph = induced_graph(train_nodes, tr, va, te)
    test_graph = induced_graph(test_nodes, tr, va, te)

    assert not train_graph.test_mask.any(), "Nodi di test nella rete di train"
    assert test_graph.test_mask.all(), "Nella rete di test devono esserci solo nodi di test"
    assert not (test_graph.train_mask.any() or test_graph.val_mask.any()), "Train o val nella rete di test"
    assert (train_graph.y != -1).all() and (test_graph.y != -1).all(), "Nodi senza etichetta nelle reti"
    assert train_graph.train_mask.sum() == tr.sum() and train_graph.val_mask.sum() == va.sum(), "Train o val incompleti"
    assert test_graph.num_nodes == te.sum(), "Test incompleto"
    train_graph.validate(raise_on_error=True)
    test_graph.validate(raise_on_error=True)

    fold_dir = os.path.join(OUT_DIR, f"fold_{k+1}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save([train_graph, test_graph], os.path.join(fold_dir, "graphs.pt"))

    test_deg = degree(test_graph.edge_index[0], test_graph.num_nodes)
    print(f"Fold {k+1}: Rete train: {train_graph.num_nodes} nodi, {train_graph.num_edges} archi "
          f"(train {tr.sum().item()}, val {va.sum().item()})    "
          f"Rete test: {test_graph.num_nodes} nodi, {test_graph.num_edges} archi, "
          f"isolati {(test_deg == 0).float().mean().item():.1%}")
