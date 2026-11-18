import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset

TUTOR_FOLDERS = {
    'r': 'percented_train_r',
    'rs': 'percented_train_rs',
    'rs50': 'percented_train_rs_percented_s_50'
}


class SocialNetwork(InMemoryDataset):
    def __init__(self, root, name='none', fold=0, transform=None, pre_transform=None):
        self.name = name
        self.fold = fold
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['real_social_network.edg', 'real_posts.tsv', 'real_users_labels.tsv']

    @property
    def processed_file_names(self):
        if self.name == 'directed':
            return ['data_directed.pt']
        elif self.name == 'directed-cv':
            return ['data_directed_cv.pt']
        elif self.name.startswith(('r-', 'rs-')):
            return [f'data_{self.name}.pt']
        elif self.name.startswith('tutor-'):
            return [f'data_{self.name}_fold{self.fold + 1}.pt']
        return ['data.pt']

    def download(self):
        raise FileNotFoundError(
            f"File raw mancanti in {self.raw_dir}: vanno copiati a mano."
        )

    def process(self):
        if self.name == 'directed':
            graph_path = osp.join(self.root, 'processed_graph_directed.pt')
        elif self.name == 'directed-cv':
            graph_path = osp.join(self.root, 'processed_graph_directed_cv.pt')
        elif self.name.startswith(('r-', 'rs-')):
            setting, percentage = self.name.split('-')
            graph_path = osp.join(self.root, 'user_classification',
                                  f'percented_train_{setting}',
                                  f'{percentage}_perc', 'graph_directed.pt')
        elif self.name.startswith('tutor-'):
            _, setting, percentage = self.name.split('-')
            graph_path = osp.join(self.root, 'user_classification', 'tutor',
                                  TUTOR_FOLDERS[setting], f'{percentage}_perc',
                                  'cross_validation', f'fold_{self.fold + 1}',
                                  'graph_directed.pt')
        else:
            graph_path = osp.join(self.root, 'processed_graph.pt')

        data = torch.load(graph_path, weights_only=False)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])