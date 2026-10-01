import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset


class SocialNetwork(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['real_social_network.edg', 'real_posts.tsv', 'real_users_labels.tsv']

    @property
    def processed_file_names(self):
        return['data.pt']

    def download(self):
        raise FileNotFoundError(
            f"File raw mancanti in {self.raw_dir}: vanno copiati a mano."
        )

    def process(self):
        graph_path = osp.join(self.root, 'processed_graph.pt')
        data = torch.load(graph_path, weights_only=False)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

