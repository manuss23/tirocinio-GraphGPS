from torch_geometric.graphgym.loader import create_dataset
from torch_geometric.loader import DataLoader

def create_inductive_loaders():
    dataset = create_dataset()
    assert len(dataset) == 2, "Il dataset induttivo deve contenere rete di train e rete di test"

    train_graph = dataset[0:1]
    test_graph = dataset[1:2]

    train_loader = DataLoader(train_graph, batch_size=1, shuffle=False)
    val_loader = DataLoader(train_graph, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_graph, batch_size=1, shuffle=False)
    return [train_loader, val_loader, test_loader]